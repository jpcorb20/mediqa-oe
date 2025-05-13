
import argparse
import json
import os
import shutil

import requests
import zipfile
import nltk

nltk.download('punkt_tab')

ACI_BENCH_URL = "https://github.com/wyim/aci-bench/archive/refs/heads/main.zip"
PRIMOCK_URL = "https://github.com/babylonhealth/primock57/archive/refs/heads/main.zip"

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_data")

def download_data(aci_bench_url, tmp_dir, name):

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        print(f"Temporary directory {tmp_dir} already exists. Skipping download.")
        return

    print(f"Downloading {name} from {aci_bench_url} to {tmp_dir}") 
        
    # download the ACI Bench zip file
    response = requests.get(aci_bench_url)
    zip_path = os.path.join(tmp_dir, name + ".zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # delete the zip file
    os.remove(zip_path)

def walk_aci_bench_directory(directory):
    """
    Walk through the directory and find all json files
    """
    transcript_data = {}
    for dirpath, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                basename = filename.replace('.json', '')
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for d in data.get("data", []):
                        src = d.get("src", None)
                        
                        file_id = "_".join(d.get("file", "").split("-")[0:2])
                        transcript_id = "acibench_" + file_id + "_" + basename
                        print(f"Processing {transcript_id} ")
                        transcript_data[transcript_id] = {
                            "transcript": src
                        }

    return transcript_data


    
def read_aci_bench_data(aci_path):
    aci_bench_json_dir_1 = os.path.join(aci_path, "aci-bench-main", "data", "challenge_data_json")
    aci_bench_json_dir_2 = os.path.join(aci_path, "aci-bench-main", "data", "src_experiment_data_json")


    # walk through the directory and find all json files
    transcript_data = {}
    print(f"Walking through {aci_bench_json_dir_1} and {aci_bench_json_dir_2}")
    transcript_data.update(walk_aci_bench_directory(aci_bench_json_dir_1))
    transcript_data.update(walk_aci_bench_directory(aci_bench_json_dir_2))

    print(f"Found {len(transcript_data)} transcripts in ACI Bench data")
    return transcript_data

def read_primock_data(primock_path):
    # run primock txtgrid_to_transcript.py
    script_path = os.path.join(primock_path, "primock57-main", "scripts", "textgrid_to_transcript.py")
    if not os.path.exists(script_path):
        print(f"Script {script_path} does not exist. Skipping Primock data.")
        return
    transcript_path = os.path.join(primock_path, "primock57-main", "transcripts")
    primock_transcript_path = os.path.join(TMP_DIR, "primock_transcript")
    os.system(f"python {script_path} --transcript_path {transcript_path} --output_path {primock_transcript_path}")
    print(f"Primock data saved to {primock_transcript_path}")

    # walk through the directory and find all json files
    transcript_data = {}
    for dirpath, _, files in os.walk(primock_transcript_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r') as f:
                    primock_id = filename.replace(".txt", "")
                    primock_id = primock_id.replace("day", "primock57_")
                    primock_id = primock_id.replace("consultation0", "")
                    primock_id = primock_id.replace("consultation", "")
                    
                    data = f.readlines()
                    data = [line.strip() for line in data if line.strip()]
                    transcript_lines = []
                    for line in data:
                        line = line.replace("Doctor:", "[doctor]")
                        line = line.replace("Patient:", "[patient]")
                        transcript_lines.append(line)
                    transcript_data[primock_id] = {
                        "transcript": "\n".join(transcript_lines)
                    }
    
    print(f"Found {len(transcript_data)} transcripts in Primock data")
    return transcript_data

def parse_transcript_str(transcript):
    turns = []
    
    turn_id = 1
    for turn in transcript.split("\n"):
        turn_transcript = turn.strip()
        if turn_transcript.startswith("[doctor]"):
            speaker = "DOCTOR"
            
        elif turn_transcript.startswith("[patient]"):
            speaker = "PATIENT"
        elif turn_transcript.startswith("[patient_guest]"):
            speaker = "PATIENT"
        else:
            print(f"Unknown speaker in transcript: {turn_transcript} using DOCTOR")
            speaker = "DOCTOR"

        # remove the speaker tag
        turn_transcript = turn_transcript.split("]", 1)[-1].strip()
        for s in nltk.sent_tokenize(turn_transcript):
            turns.append({
                "turn_id": turn_id,
                "speaker": speaker,
                "transcript": s
            })
            turn_id += 1

    return turns

def attach_transcript_section(section, transcript_dict):
    for d in section:
        id = d.get("id", None)  
        if id in transcript_dict:
            transcript = transcript_dict[id]["transcript"]
            d["transcript"] = parse_transcript_str(transcript)
        else:
            print(f"Transcript for {id} not found in ACI Bench or Primock data")

def attach_transcript(input_file, output_file, transcript_dict):
    with open(input_file, 'r') as f:
        original_data = json.load(f)

    train_section = original_data.get("train", [])
    dev_section = original_data.get("dev", [])
    test_section = original_data.get("test", [])
    
    # attach transcript to train, dev and test sections
    attach_transcript_section(train_section, transcript_dict)
    attach_transcript_section(dev_section, transcript_dict)
    attach_transcript_section(test_section, transcript_dict)

        
    with open(output_file, 'w') as f:
        json.dump(original_data, f, indent=4)
        
def main(aci_bench_url, primock_url, input_file, output_file, cleanup):


    if cleanup:
        print("Cleaning up temporary files...")
        try:
            shutil.rmtree(TMP_DIR)
            print(f"Removed {TMP_DIR}")
        except Exception as e:
            print(f"Error removing {TMP_DIR}: {e}")

        print("Run the script again to download the data and add transcripts.")
        return

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    aci_bench_tmp_dir = os.path.join(TMP_DIR, "aci-bench")
    primock_tmp_dir = os.path.join(TMP_DIR, "primock")

    # download ACI Bench data
    download_data(aci_bench_url, aci_bench_tmp_dir, "aci-bench")
    # download Primock data
    download_data(primock_url, primock_tmp_dir, "primock")

    transcript_dict = {}
    # read ACI Bench data
    transcript_dict.update(read_aci_bench_data(aci_bench_tmp_dir))

    # read Primock data
    transcript_dict.update(read_primock_data(primock_tmp_dir))

    #print keys of transcript_dict
    print(f"Found {len(transcript_dict.keys())} transcripts in total")
    
    attach_transcript(input_file, output_file, transcript_dict)
    print(f"Attached transcripts saved to {output_file}")



    
    print(f"Temporary files saved to {TMP_DIR}. Use --cleanup to remove them.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process OE data to attach transcript")
    argparser.add_argument("--aci-bench-url", type=str, default=ACI_BENCH_URL, help="ACI Bench data URL")
    argparser.add_argument("--primock-url", type=str, default=PRIMOCK_URL, help="Primock data URL")
    argparser.add_argument("--input-file", type=str, default="data/orders_data.json", help="Input json file")
    argparser.add_argument("--output-file", type=str, default="data/orders_data_transcript.json", help="Output json file")
    argparser.add_argument("--cleanup", action="store_true", help="Cleanup temporary files")
    
    args = argparser.parse_args()
    main(args.aci_bench_url, args.primock_url, args.input_file, args.output_file, args.cleanup)
import os
import json
import argparse
from tqdm import tqdm
from openai import AzureOpenAI, APIConnectionError

from evaluation.order import Order

from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    ChainedTokenCredential,
    get_bearer_token_provider,
)

TRIGGER_MESSAGE = "---ORDER EXTRACTION---"
DEFAULT_API_VERSION = "2025-01-01-preview"
DEFAULT_PTOMPT_PATH = "extraction/basic_prompt.txt"
DEFAULT_DATA_PATH = "data/orders_data_transcript.json"
DEFAULT_OUT_PATH = "results/generated_orders.json"
DEFAULT_EXECUTION_SETTINGS = {
    "max_tokens": 4320,
    "temperature": 0.0,
    "top_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "n": 1,
}
DEFAULT_REASONING_EXECUTION_SETTINGS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "n": 1,
    "max_completion_tokens": 24320,  # for reasoning models
}

def get_token_credential():
    chained_credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(),
    )
    return chained_credential


def get_aoai_model_response(client, prompt, transcript, model="gpt-4o"):
    # build the messages
    
    if model.startswith("gpt-4"):
        settings = DEFAULT_EXECUTION_SETTINGS.copy()
        role = "system"
    else:
        settings = DEFAULT_REASONING_EXECUTION_SETTINGS.copy()
        role = "user"
    messages = [
            {"role": role, "content": f"{prompt}\n{transcript}"},
            {"role": role, "content": f"{TRIGGER_MESSAGE}\n"},
        ]

    # call the openai client
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **settings
        )
    except APIConnectionError as e:
        return None, True

    return response, False


def get_orders_from_model_response(response):
    raw_response = response.choices[0].message.content
    raw_response = raw_response.strip()
    raw_response = raw_response.replace("```json", "")
    raw_response = raw_response.replace("```", "")

    # get json from the response
    try:
        json_message = json.loads(raw_response) 
    except json.JSONDecodeError:
        #print(f"Error decoding JSON for file {fname}.")
        #print(f"Response content:\n {raw_response}")
        return None, True

    # parse json message to order objects
    orders_parsed = []
    for order in json_message:
        try:
            order_obj = Order.from_dict(order)
            orders_parsed.append(order_obj)
        except:
            print(f"Error creating order in {fname}:\n {order}")
            return None, True
    
    return orders_parsed, False

def write_orders_to_file(orders, output_path, runid=None):
    # check if the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if runid is provided, append it to the output filename
    if runid is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_runid{runid}{ext}"
    
    with open(output_path, 'w') as f:
        f.write(json.dumps(orders, indent=4))

def get_text_from_turns(turns):
    speaker_turns = [f"[{turn['speaker']}] {turn['transcript']}" for turn in sorted(turns, key=lambda x: x['turn_id'])]
    return "\n".join(speaker_turns)

def load_transcripts(input_path, dataset="dev"):
    # check if the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    transcripts = {}
    with open(input_path, 'r') as f:
        data = json.loads(f.read())

    fail_transcripts = []
    for elem in data[dataset]:
        transcript_id = elem["id"]
        try:
            turns = elem["transcript"]
            transcript_text = get_text_from_turns(turns)
            transcripts[transcript_id] = transcript_text
        except KeyError as e:
            fail_transcripts.append(transcript_id)

    print(f"loaded {len(transcripts)} transcripts")
    print(f"failed to load {len(fail_transcripts)} transcripts")
    return transcripts

def get_aoai_client(endpoint, deployment_name):
    # create the openai client
    token_provider = get_bearer_token_provider(get_token_credential(), "https://cognitiveservices.azure.com/.default")

    # create openai client using the token provider
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=DEFAULT_API_VERSION,
        azure_deployment=deployment_name,
    )

    return client

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Basic code to run order extraction from transcript")
    argparser.add_argument("--input_path", type=str, help="Path to the input file containing the transcripts", default=DEFAULT_DATA_PATH)
    argparser.add_argument("--output_path", type=str, help="Path to the output file to save the extracted orders", default=DEFAULT_OUT_PATH)
    argparser.add_argument("--prompt_path", type=str, help="Path to the prompt file", default=DEFAULT_PTOMPT_PATH)
    argparser.add_argument("--endpoint", type=str, help="Azure OpenAI endpoint", required=True)
    argparser.add_argument("--deployment_name", type=str, help="Azure OpenAI deployment name", required=True)
    argparser.add_argument("--model", type=str, help="Model to use for order extraction", default="gpt-4o")
    argparser.add_argument("--dataset", type=str, help="train or dev", default="dev")
    argparser.add_argument("--runids", type=int, nargs='+', default=[1], help="List of seeds to use for random operations")
    args = argparser.parse_args()

    # load the prompt and transcripts:
    transcripts = load_transcripts(args.input_path, args.dataset)
    with open(args.prompt_path, 'r') as f:
        prompt = f.read()

    # get the openai client
    client = get_aoai_client(args.endpoint, args.deployment_name)

    # run the order extraction
    for runid in args.runids:
        print(f"#### Running order extraction for runid {runid}...")

        all_orders = {}
        num_errs_model = 0
        num_errs_parsing = 0
        for fname in tqdm(transcripts.keys()):
            #print(f"Processing transcript {fname}...")

            transcript = transcripts[fname]
            response, error = get_aoai_model_response(client, prompt, transcript, model=args.model)

            if error:
                #print(f"  --> Error getting model response for transcript {fname}")
                num_errs_model += 1
                orders_parsed = []

            else:
                orders_parsed, error = get_orders_from_model_response(response)

                if error:
                    orders_parsed = []
                    num_errs_parsing += 1
                    #print(f"  --> Error parsing response in transcript {fname}")

            all_orders[fname] = [order.to_dict() for order in orders_parsed]

        write_orders_to_file(all_orders, args.output_path, runid=runid)
        print(f"#### Finished order extraction for runid {runid}.")
        print(f"  --> Number of errors in model response: {num_errs_model}")
        print(f"  --> Number of errors in parsing response: {num_errs_parsing}")
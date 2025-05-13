# mediqa-oe
Codebase for Order Extraction Tasks as part of MEDIQA-OE @ ClinicalNLP (COLM 2025)


# Data Statistic

# Data format description

# Data processing

The data only contains the annotation for orders. You need to run the `process_data` script to attach the transcript to each encounter.
It's recommended to use a [virtual environement](https://docs.python.org/3/library/venv.html) to install pythons packages. 

```
process_data.py --help
usage: process_data.py [-h] [--aci-bench-url ACI_BENCH_URL] [--primock-url PRIMOCK_URL] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--cleanup]

Process OE data to attach transcript

options:
  -h, --help            show this help message and exit
  --aci-bench-url ACI_BENCH_URL
                        ACI Bench data URL
  --primock-url PRIMOCK_URL
                        Primock data URL
  --input-file INPUT_FILE
                        Input json file
  --output-file OUTPUT_FILE
                        Output json file
  --cleanup             Cleanup temporary files
```
Example:
```bash
$ pip install -r requirements.txt
$ python data/process_data.py
```

The script will generate the file `data/orders_data_transcript.json` 

## License
The data here is published under Community Data License Agreement - Permissive - Version 2.0 https://cdla.dev/permissive-2-0/

## Issue
If you encounter an issue with attaching transcript. try running the `process_data.py` script with the `--cleanup` flag to remove the downloaded files.
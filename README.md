S# MEDIQA-OE: Medical Order Extraction from Doctor-Patient Dialogs
Codebase for Order Extraction Tasks as part of MEDIQA-OE @ ClinicalNLP 2025


# Data Statistic

| Dataset | # Encounters | Follow-up | Imaging | Lab | Medication | Total Orders |
|---------|--------------|-----------|---------|-----|------------|---------------|
| Train   | 63           | 25        | 14      | 29  | 75         | 143           |
| Dev     | 100          | 41        | 26      | 71  | 117        | 255           |

# Data format description

The provided dataset is a JSON file with the following structure:

	{
	  "train": [
		  {
			  "id": ...,
			  "expected_orders":  [...]
		  }
	  ],
	  "dev": [...]
	}

Orders contain four keys: order_type, description, reason, and provenance. After attaching the transcripts from the data sources (Primock57 and ACI-Bench), it is

	{
	  "train": [
		  {
			  "id": ...,
			  "expected_orders":  [...],
			  "transcript": [
				  {
					  "turn_id": 0,
					  "speaker": "DOCTOR",
					  "transcript": "..."
				  },
				  ...
			  ]
		  }
	  ],
	  "dev": [...]
	}

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

# Evaluation

Check out `evaluation/README.md` for more information on how to run local evaluation.

# License
The data here is published under Community Data License Agreement - Permissive - Version 2.0 https://cdla.dev/permissive-2-0/

# Issue
If you encounter an issue with attaching transcript. try running the `process_data.py` script with the `--cleanup` flag to remove the downloaded files.

# Cite

	@article{corbeil2025empowering,
	  title={Empowering Healthcare Practitioners with Language Models: Structuring Speech Transcripts in Two Real-World Clinical Applications},
	  author={Corbeil, Jean-Philippe and Abacha, Asma Ben and Michalopoulos, George and Swazinna, Phillip and Del-Agua, Miguel and Tremblay, Jerome and Daniel, Akila Jeeson and Bader, Cari and Cho, Yu-Cheng and Krishnan, Pooja and others},
	  journal={arXiv preprint arXiv:2507.05517},
	  year={2025}
	}

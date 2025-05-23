# MEDIQA-OE Evaluation Tool

Here's an example of evaluation execution:

    python evaluate_oe.py -t truth_orders.json -p pred_orders.json -o output_dir

where `truth_orders.json` and `pred_orders.json` contain each a JSON object using transcript identifiers as keys, and associated values are JSON array containing JSON objects that are the orders (keys: `description`, `reason`, `order_type` and `provenance`).
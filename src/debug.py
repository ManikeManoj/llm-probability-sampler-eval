from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
)

def show_tokenization(text, label=None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    decoded_parts = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in ids
    ]
    decoded_full = tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    if label:
        print(f"\n--- {label} ---")

    print(f"text        : {text!r}")
    print(f"ids         : {ids}")
    print(f"pieces      : {pieces}")
    print(f"decoded_each: {decoded_parts}")
    print(f"decoded_full: {decoded_full!r}")


single_chars = [
    "-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

multi_digit_strings = [
    "00", "01", "10", "11", "12", "20", "42", "99",
    "100", "123", "456", "999",
    "0.1", "0.12", "0.123",
    "4.0", "4.12", "4.123",
    "-1", "-12", "-1.2", "-12.34",
]

prefixes = [
    "",
    "Answer: ",
    "The number is ",
    "4",
    "4.",
    "4.1",
    "-",
    "-0.",
    "-12.",
]

print("\n==============================")
print("SINGLE CHARACTER TOKENIZATION")
print("==============================")

for s in single_chars:
    show_tokenization(s, label=s)


print("\n==============================")
print("MULTI-DIGIT / NUMBER TOKENIZATION")
print("==============================")

for s in multi_digit_strings:
    show_tokenization(s, label=s)


print("\n==============================")
print("PREFIX + CANDIDATE TOKENIZATION")
print("==============================")

candidates = single_chars + [
    "10", "12", "100", "123", "0.1", "4.2", "-1"
]

for prefix in prefixes:
    print("\n" + "=" * 80)
    print(f"PREFIX: {prefix!r}")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    print(f"prefix_ids: {prefix_ids}")
    print(f"prefix_pieces: {tokenizer.convert_ids_to_tokens(prefix_ids)}")

    for cand in candidates:
        full = prefix + cand
        full_ids = tokenizer.encode(full, add_special_tokens=False)

        if full_ids[:len(prefix_ids)] == prefix_ids:
            tail_ids = full_ids[len(prefix_ids):]
            status = "clean-tail"
        else:
            tail_ids = full_ids
            status = "retokenized"

        tail_pieces = tokenizer.convert_ids_to_tokens(tail_ids)
        tail_decoded = [
            tokenizer.decode([i], clean_up_tokenization_spaces=False)
            for i in tail_ids
        ]

        print(
            f"  + {cand!r:<6} | {status:<11} | "
            f"tail_ids={tail_ids} | pieces={tail_pieces} | decoded={tail_decoded}"
        )


print("\n==============================")
print("STEP-BY-STEP DIGIT GROWTH")
print("==============================")

growth_examples = [
    "0",
    "01",
    "012",
    "0123",
    "4",
    "42",
    "421",
    "4210",
    "4.1",
    "4.12",
    "4.123",
    "-1",
    "-12",
    "-123",
    "-12.3",
    "-12.34",
]

for s in growth_examples:
    show_tokenization(s, label=s)
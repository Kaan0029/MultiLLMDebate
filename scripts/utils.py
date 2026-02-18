# utils.py
import json
import re

def safe_parse_json(text):
    """
    Robust JSON extractor – handles poorly formatted LLM outputs.
    """
    try:
        return json.loads(text)
    except:
        # attempt naive extraction
        try:
            match = re.search(r"{.*}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass

    return {"label": "UNK", "reason": text}


# def extract_label(parsed):
#     """
#     Extract label from parsed JSON (Round 1 or Round 2).
#     """
#     for key in ["label", "updated_label"]:
#         if key in parsed:
#             return parsed[key]

#     return "UNK"


def extract_label(parsed):
    """
    Extract label from parsed JSON (Round 1, Round 2, or Referee).
    """
    for key in ["label", "updated_label", "final_label"]:
        if key in parsed:
            return parsed[key]
    return "UNK"


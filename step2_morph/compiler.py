# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Language compiler: prompt -> spec (behaviour objective + morphology constraints).

A local LLM (Ollama, qwen3.6) reads the prompt and the robot API below and
returns JSON. The spec is validated (known descriptors, known operators,
values inside the design space) before any evolution uses it; nothing is
hand-edited afterwards.

Usage: python compiler.py            compiles every prompt in prompts.json -> specs/<name>.json
"""

import json
import os
import time
import urllib.error
import urllib.request

import morph

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "qwen3.6:latest"
OLLAMA = "http://localhost:11434/api/generate"

API = f"""You translate a natural-language request for a legged robot into a JSON spec
that a robot-design optimiser will use. Reply with JSON only.

The robot: a box torso with mirrored leg pairs; each joint is a hobby servo.
World axes: +x = the robot's forward, +y = the robot's left, yaw > 0 = turning left (counter-clockwise).
An episode lasts 5.5 s.

BEHAVIOUR descriptors (measured on the episode; the optimiser MAXIMISES sum(weight * descriptor)):
  dx       forward displacement in metres (negative = backward)
  dy       leftward displacement in metres (negative = rightward)
  yaw      total turning in radians (positive = left / counter-clockwise)
  airtime  fraction of time with no part of the robot touching the ground (0..1)
  abs_dx, abs_dy, abs_yaw   absolute values of the above
  travel   straight-line distance travelled in metres
Typical good values: dx about 1-3 m, yaw about 5-20 rad, airtime about 0.1-0.3.
Use negative weights to penalise unwanted motion (e.g. turning on the spot: reward yaw, penalise travel).

MORPHOLOGY descriptors (fixed by the body) with their possible ranges:
  n_legs        2, 4 or 6
  n_joints      2 .. 18
  mean_leg_len  {3 * 0 + morph.SEG_RANGE[0]:.2f} .. {3 * morph.SEG_RANGE[1]:.2f} m (sum of segment lengths, averaged over legs)
  max_leg_len   same range
  torso_L       {morph.L_RANGE[0]:.2f} .. {morph.L_RANGE[1]:.2f} m (length along x)
  torso_W       {morph.W_RANGE[0]:.2f} .. {morph.W_RANGE[1]:.2f} m (width along y)
  mass          about 0.9 .. 2.6 kg
  stand_z       torso height when standing, about 0.10 .. 0.50 m
Constraints use op "==", ">=" or "<=". Only add a morphology constraint when the request
says something about the body; otherwise use an empty list.

JSON format:
{{"behaviour": [{{"descriptor": "...", "weight": number}}, ...],
  "morphology": [{{"descriptor": "...", "op": "==|>=|<=", "value": number}}, ...],
  "reasoning": "one short sentence"}}
"""


def schema():
    from evolve_morph import BEHAVIOUR_DESC, MORPH_DESC
    return {"type": "object", "required": ["behaviour", "morphology", "reasoning"], "properties": {
        "behaviour": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["descriptor", "weight"],
                      "properties": {"descriptor": {"type": "string", "enum": BEHAVIOUR_DESC},
                                     "weight": {"type": "number"}}}},
        "morphology": {"type": "array", "items": {"type": "object", "required": ["descriptor", "op", "value"],
                       "properties": {"descriptor": {"type": "string", "enum": MORPH_DESC},
                                      "op": {"type": "string", "enum": ["==", ">=", "<="]},
                                      "value": {"type": "number"}}}},
        "reasoning": {"type": "string"}}}


def ask(prompt, tries=3):
    """Structured output: Ollama constrains the reply to the JSON schema above."""
    body = json.dumps({"model": MODEL, "prompt": API + "\nRequest: " + prompt, "format": schema(),
                       "stream": False, "think": False, "options": {"temperature": 0}}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            return json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["response"])
        except (urllib.error.HTTPError, json.JSONDecodeError) as e:
            if k == tries - 1:
                raise
            print("    retry after", type(e).__name__)
            time.sleep(5)


def validate(spec):
    from evolve_morph import BEHAVIOUR_DESC, MORPH_DESC
    errs = []
    if not spec.get("behaviour"):
        errs.append("no behaviour terms")
    for t in spec.get("behaviour", []):
        if t.get("descriptor") not in BEHAVIOUR_DESC or not isinstance(t.get("weight"), (int, float)):
            errs.append(f"bad behaviour term {t}")
    for c in spec.get("morphology", []):
        if c.get("descriptor") not in MORPH_DESC or c.get("op") not in ("==", ">=", "<=") \
                or not isinstance(c.get("value"), (int, float)):
            errs.append(f"bad morphology constraint {c}")
    return errs


def main():
    prompts = json.load(open(os.path.join(HERE, "prompts.json"), encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "specs"), exist_ok=True)
    for group in ("A", "B"):
        for name, text in prompts[group].items():
            spec, attempts = ask(text), []
            errs = validate(spec)
            while errs and len(attempts) < 2:          # send the errors back once or twice
                attempts.append({"spec": spec, "errors": errs})
                spec = ask(text + "\nYour previous JSON was invalid: " + "; ".join(errs) +
                           ". Every morphology constraint needs descriptor, op and a numeric value.")
                errs = validate(spec)
            spec = {"name": name, "prompt": text, "compiler": MODEL, **spec, "validation_errors": errs,
                    "repair_attempts": attempts}
            json.dump(spec, open(os.path.join(HERE, "specs", f"{name}.json"), "w"), indent=1)
            beh = " + ".join(f"{t.get('weight')}*{t.get('descriptor')}" for t in spec.get("behaviour", []))
            mor = ", ".join(f"{c.get('descriptor')} {c.get('op')} {c.get('value')}" for c in spec.get("morphology", []))
            print(f"{name:16s} \"{text}\"\n    maximise {beh}\n    subject to [{mor}]  {('ERRORS ' + str(errs)) if errs else ''}"
                  f"{'  (repaired after ' + str(len(attempts)) + ' attempt)' if attempts else ''}")


if __name__ == "__main__":
    main()

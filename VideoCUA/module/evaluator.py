
TRAJECTORY_EVAL_FORMAT_PROMPT = """
# Task:
{goal}

# Steps:
{steps}
"""

FINAL_TRAJECTORY_EVAL_PROMPT = """
An computer-use agent has completed a multi-step computer-use task. Below are all the steps, with each step's action, computer state changes that the actions made, judgement, and reflection. Now you must produce a single, final evaluation of how well the entire task was executed from start to end.

The final output MUST be exactly one valid JSON object, with these keys:

1) "task_completed": a boolean indicating if the main goal was fully completed or not. There will always be a terminate action at the end of the task, do not be affected by it.
2) "alignment_score": an integer from 0 to 10 measuring how closely the actions and final state align with the original goal:
   - 0 means it barely matched the goal or failed entirely.
   - 10 means a perfect match.
3) "efficiency_score": an integer from 0 to 10 measuring how efficient the trajectory was:
   - 0 means many redundant or incorrect steps, or extremely roundabout, or many repeated steps.
   - 10 means minimal or no wasted steps.
4) "task_difficulty": an integer from 0 to 10 representing how difficult or complex this task is, regardless of whether it was completed well. 0 means trivial, 10 means very complex.
5) "reason": a brief textual summary explaining your final reasoning. Summarize any large errors or highlight good aspects. This can mention whether steps were repeated, whether too many mistakes were made, or if it's a near-perfect run.
6) "natural_language_task": a more natural way to describe the task and actions in a few sentences. This should be a more human-friendly version of the task and actions as if it is asked by a human user. Both question format and statement format are acceptable. Necessary setting, details and parameters should be included to make sure the task aligns with the computer state and the correct actions (If without any parameter, the model won't be able to take the actions, the parameter should be added).
7) "actual_task": according to the original task and the steps, what was the actual task being performed? Ignore the incorrect and redundant steps. This should be not a verbatim copy of the original task description. Necessary setting, details, texts, messages and parameters should be included to make sure the task aligns with the computer state and the correct actions (If without any parameter, the model won't be able to take the actions, the parameter should be added). Describe the task using an imperative command, as if the user is instructing the computer to perform it.

No extra keys. No text outside the JSON. No Markdown. Do not re-list all the steps. Only produce this final JSON object.

{
  "task_completed": bool,
  "alignment_score": int,
  "efficiency_score": int,
  "reason": str,
  "actual_task": str,
  "natural_language_task": str,
  "task_difficulty": int
}
""".strip()

STEP_FORMAT = """## Step {index}
### Thought:
{thought}
### Action:
{action}
### Is the current step correct?
{last_step_correct}
### Is the current step redundant?
{last_step_redundant}
### The action made the following changes to the computer state:
{reflection}
"""


def generate_traj_eval_history(generated_steps):
    history = ""
    for i, step in enumerate(generated_steps):
        history += STEP_FORMAT.format(
            index=i+1,
            thought=step['value']['thought'],
            action=step['value']['action'],
            last_step_correct=step['value']['last_step_correct'],
            last_step_redundant=step['value']['last_step_redundant'],
            reflection=step['value']['reflection']
        )

    return history

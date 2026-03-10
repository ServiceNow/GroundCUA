import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError


class GeneratorResponse(BaseModel):
    observation: str = Field(..., description="Detailed description of the current screen and context.")
    thought: str = Field(..., description="Agent's reasoning about the current situation.")
    action: str = Field(..., description="The action the agent intends to take.")


COT_GENERATOR_PROMPT_FOR_MOUSE_ACTION = """Your task is to generate three components: `Observation`, `Thought`, and `Action`, for some computer-use actions. These components must reason about the current state of the task based on the history and current screenshot, evaluate task progress, and guide the model to predict the "Predicted code". Avoid directly referencing the provided predicted action or any visual aids (e.g., red circles, dots or image patches).

You are provided with the following information:
1. **Previous Actions**: A list of actions that have been taken so far. It is given in the form of pyautogui
2 **Former thought**: A description of the thought process of the previous action.
3. **Effect of former action**: A description of the effect of the previous action on the computer state.
4. **Goal**: A description of the task to be accomplished.
5. **Predicted code**: The next action to be predicted, which includes the type of action and the specific arguments required to execute it. It is given in the form of pyautogui
6. **Full Screenshot**: A screenshot showing the computer's current state.
7. **Image Patch**: A cropped image centered around the coordinate of the predicted action.

**Previous Actions**:
{previous_actions}

**Former thought**:
{former_thought}

**Effect of former action**:
{former_action_effect}

**Goal**:
{goal}

**Predicted code**:
{action_commands}

### **Output**:
You must generate the following three components:

1. **Observation**:
    - Describe the necessary current computer state based on the current full screenshot in detail.
    - Any content, element, options, information or clues that are possibly relevant to achieving the task goal. Describe their name, content, location or shape (if possible).
    - Application Context: active application, active window or page, overall layout.
    - Any elements that may affect the task execution: pop-ups, notifications, error messages, loading states, etc.
    - Cover the elements near the red circle.
    - If the element name or text is not in English, no need to translate it.
    - DO NOT mention the image patch, red circle, red dot, or mouse position in any part of the output in the observation!

2. **Thought**:
    - State changes:
        - Adjust "Effect of former action" based on the current screenshot and ADD it to the beginning of Thought. Naturally connect it with the rest of the thought.
        - This reflection should be naturally integrated into the thought process, **as if you're thinking aloud**, not directly referring to the "Effect of former action" or "reflection".
    - Memory:
        - Add necessary information according to the history, former thought and current screenshot.
    - Step by Step assess the progress towards completing the task:
        - Analyze what parts of the task have already been completed and how they contribute to the overall goal.
        - Make a plan or adjust the former plan on how to complete the task based on the history and currect screenshot.
    - **Propose the logical next action**:
        - List possible next actions that could logically follow from the current state to progress toward the goal.
        - Evaluate these possible actions based on the current state and previous actions.
        - Carefully examine the element at the center of the red circle in both images. It is the actual clicking position, so it is the action to be proposed. Explain why this action is the most logical and likely choice among the alternatives.
        - The logical next action should match the current full screenshot and the action types in 'Predicted code'.
        - Anticipate the consequences of the logical next action (how the computer state will likely change after the action is performed).
    - You can only say the logical next action, never mention 'predicted action'. Do NOT say expressions such as the 'predicted action' suggests that... or intends to ... Pretend you do not know the 'predicted action'!
    - DO NOT mention the image patch, red circle, red dot or mouse position in any part of the output in the thought!
    - Use the first-person perspective to represent the thought process.

3. **Action**:
    Provide a clear, actionable instruction based strictly on the `Predicted code`. Ensure:
        - The instruction MUST aligns exactly with the `Predicted code` and the current full screenshot, describing `Predicted code` in a concise and actionable manner. No deviations or assumptions about alternative actions are allowed. **Avoid** vague descriptions like "indicated in the image", "relevant" or any personal preference-based phrases.
        - Note: pyautogui.moveTo(A) and pyautogui.dragTo(B) together means drag from A to B, pyautogui.moveTo(A) and pyautogui.scroll together means scroll the mouse wheel at A.
        - Carefully examine the element at the center of the red circle in the image patch.
    - If the action involves interacting with a specific target (e.g., clicking, dragging), describe the target explicitly. Avoid directly using coordinates.
        - Focus on the red circle's exact center to determine which button or element is being targeted. **Do NOT use the mouse pointer position** to infer the target.
        - When clicking an element like an icon, specify the element's name whenever possible. If the element's name is not identifiable, describe its features like shape, color, position or relationship to other elements.
        - When interacting with buttons in the top-right corner of an application window (e.g., minimize, maximize, close), ensure the target button is correctly identified.
        - If clicking on a specific portion of text, describe where exactly the click occurred within the text.
        - If the click position corresponds to an empty space, it may serve purposes such as closing a pop-up, refocusing a window, or dismissing a modal.

### **Important Notes**:
1. You are provided with two images of the current computer state:
    - The full screenshot, which gives you the overall context of the window and its contents.
    - The image patch, which is centered at the predicted click position and corresponds to the provided `xy` coordinates. The image patch may be cropped if the click position is near the edge of the screen.
  - Both images highlight the click position with a red circle, and the exact center of the circle contains a red dot, marking the target for click-related actions.
  - The `xy` coordinates are normalized to 0 and 1, representing the relative position on the screen, scaled to the screen size.
  - AVOID using the red circle as reasoning support, and DO NOT mention the red circle in any part, as it represents hindsight rather than predictive insight.

2. Your principles should assume that another model will not be provided with the image patch, red circle, red dot, or predicted action. Your task is to guide this model to learn to generate the `observation`, `thought`, `instruction` and predicted action based solely on the 'Goal', 'Previous Actions', and full screenshot.

3. For mouse-related actions, ignore the position of the mouse pointer in the screenshot. Do NOT mention where the mouse is. **Do NOT use the mouse position** to infer any target. The mouse position is unrelated to reasoning and does not provide any hints about the task.

4. **Extremely important**: **DO NOT** mention the provided predicted action, mouse position, highlighted buttons, red circles, red dots or image patch in any part of the response.

Respond in strict accordance with the required format. No extra text, no additional sections beyond this structure:
## Observation:
observation

## Thought:
thought

## Action:
action
""".strip()



COT_GENERATOR_PROMPT_FOR_KEYBOARD_ACTION = """Your task is to generate three components: `Observation`, `Thought`, and `Action`, for some computer-use actions. These components must reason about the current state of the task based on the history and current screenshot, evaluate task progress, and guide the model to predict the "Predicted code". Avoid directly referencing the provided predicted action or any visual aids (e.g., red circles, dots or image patches).

You are provided with the following information:
1. **Previous Actions**: A list of actions that have been taken so far. It is given in the form of pyautogui
2 **Former thought**: A description of the thought process of the previous action.
3. **Effect of former action**: A description of the effect of the previous action on the computer state.
4. **Goal**: A description of the task to be accomplished.
5. **Predicted code**: The next action to be predicted, which includes the type of action and the specific arguments required to execute it. It is given in the form of pyautogui
6. **Full Screenshot**: A screenshot showing the computer's current state.

**Previous Actions**:
{previous_actions}

**Former thought**:
{former_thought}

**Effect of former action**:
{former_action_effect}

**Goal**:
{goal}

**Predicted code**:
{action_commands}

### **Output**:
You must generate the following three components:

1. **Observation**:
    - Describe the necessary current computer state based on the current full screenshot in detail.
    - Any content, element, options, information or clues that are possibly relevant to achieving the task goal. Describe their name, content, location or shape (if possible).
    - Application Context: active application, active window or page, overall layout.
    - Any elements that may affect the task execution: pop-ups, notifications, error messages, loading states, etc.

2. **Thought**:
    - State changes:
        - Adjust "Effect of former action" based on the current screenshot and ADD it to the beginning of Thought. Naturally connect it with the rest of the thought.
        - This reflection should be naturally integrated into the thought process, **as if you're thinking aloud**, not directly referring to the "Effect of former action" or "reflection".
    - Memory:
        - Add necessary information according to the history, former thought and current screenshot.
    - Step by Step assess the progress towards completing the task:
        - Analyze what parts of the task have already been completed and how they contribute to the overall goal.
        - Make a plan or adjust the former plan on how to complete the task based on the history and currect screenshot.
    - Propose the logical next action:
        - List possible next actions that could logically follow from the current state to progress toward the goal.
        - Evaluate these possible actions based on the current state and previous actions.
        - Since the actual action is provided, explain why this action is the most logical and likely choice among the alternatives.
        - Anticipate the consequences of the logical next action (how the computer state will likely change after the action is performed).
    - For text editing or input-related actions:
        - Observe the current cursor position to understand where the user intends to input or modify text.
        - Consolidate repetitive actions (e.g., multiple spaces, backspaces, delete, or enter) into a single description, specifying how many times the action was performed.
        - Reason about the user's intent and predict what the final text should look like after the action.
        - Ensure the instruction reflects the desired text state rather than the intermediate keystrokes.
    - You can only say the logical next action, never mention 'predicted action'. Do NOT say expressions such as the 'predicted action' suggests that... or intends to ... Pretend you do not know the 'predicted action'.
    - Use the first-person perspective to represent the thought process of the annotator solving the task.

3. **Action**:
    Provide a clear, actionable instruction based strictly on the `Predicted code`. Ensure:
    - The instruction MUST aligns exactly with the `Predicted code`, describing `Predicted code` in a concise and actionable manner. No deviations or assumptions about alternative actions are allowed. **Avoid** vague descriptions like "indicated in the image", "relevant" or any personal preference-based phrases.
    - If the action involves `press` or `write`, infer the user's intended operation and action based on the `keys` provided and the cursor's current position. Use this context to reason about what the user is trying to achieve.
        - Include the action description and the desired result of the typing action and ensure the instruction reflects the intended text outcome, not just intermediate keystrokes.
        - If it's 'write' action, the instruction should first exactly matches the 'content' of action.

**Important Note**:
1. You are provided with the current **full screenshot**, which gives you the overall context of the current window and its contents.

2. Your principles should assume that another model will not be provided with the predicted action. Your task is to guide this model to learn to generate the `observation`, `thought`, `instruction` and predicted action based solely on the 'Goal', 'Previous Actions', and full screenshot.

3. The output should include only logical and actionable instructions based on the goal, task context, and action history, without referencing any predicted actions.

Respond in strict accordance with the required format. No extra text, no additional sections beyond this structure:
## Observation:
observation

## Thought:
thought

## Action:
action
""".strip()


REFLECT_COT_GENERATOR_PROMPT_FOR_MOUSE_ACTION = """Your task is to generate three components: `Observation`, `Thought`, and `Action`, for some computer-use actions. These components must reason about the current state of the task based on the history and current screenshot, evaluate task progress, and guide the model to predict the "Predicted code". Avoid directly referencing the provided predicted action or any visual aids (e.g., red circles, dots or image patches).

You are provided with the following information:
1. **Previous Actions**: A list of actions that have been taken so far. It is given in the form of pyautogui
2. **Former thought**: A description of the thought process of the previous action.
3. **Reflection of the former incorrect or redundant action**: The reason why the former action was wrong or redundant.
4. **Goal**: A description of the task to be accomplished.
5. **Predicted code**: The next action to be predicted, which includes the type of action and the specific arguments required to execute it. It is given in the form of pyautogui
6. **Full Screenshot**: A screenshot showing the computer's current state.
7. **Image Patch**: A cropped image centered around the coordinate of the predicted action.

**Previous Actions**:
{previous_actions}

**Former thought**:
{former_thought}

**Reflection of the former incorrect or redundant action**:
{former_action_effect}

**Goal**:
{goal}

**Predicted code**:
{action_commands}

### **Output**:
You must generate the following three components:

1. **Observation**:
    - Describe the necessary current computer state based on the current full screenshot in detail.
    - Any content, element, options, information or clues that are possibly relevant to achieving the task goal. Describe their name, content, location or shape (if possible).
    - Application Context: active application, active window or page, overall layout.
    - Any elements that may affect the task execution: pop-ups, notifications, error messages, loading states, etc.
    - Cover the elements near the red circle.
    - If the element name or text is not in English, no need to translate it.
    - DO NOT mention the image patch, red circle, red dot, or mouse position in any part of the output in the observation!

2. **Thought**:
    - Reflect on the former incorrect action:
        - Clearly explain why the former action was wrong or redundant based on the current screenshot and the "Reflection of the former incorrect or redundant action".
        - Naturally ADD the reflection at the beginning of Thought.
        - The current Action and Predicted code will be an atempt to correct the former action.
        - This reflection should be naturally integrated into the thought process, **as if you're thinking aloud**, not directly referring to the "Effect of former action" or "reflection".
    - Memory:
        - Add necessary information according to the history, former thought and current screenshot.
    - Step by Step assess the progress towards completing the task:
        - Analyze what parts of the task have already been completed and how they contribute to the overall goal.
        - Adjust the former plan given the former action was incorrect or redundant based on the history and currect screenshot.
    - Propose the logical next action to correct the former mistake and explain why:
        - Carefully examine the element at the center of the red circle in both images. It is the actual clicking position, so it is the action to be proposed. Explain why this action is the most logical and likely choice among the alternatives.
        - The logical next action should match the current full screenshot and the action types in 'Predicted code'.
        - Anticipate the consequences of the logical next action (how the computer state will likely change after the action is performed).
    - You can only say the logical next action, never mention 'predicted action'. Do NOT say expressions such as the 'predicted action' suggests that... or intends to ... Pretend you do not know the 'predicted action'!
    - DO NOT mention the image patch, red circle, red dot or mouse position in any part of the output in the thought!
    - Use the first-person perspective to represent the thought process.

3. **Action**:
    Provide a clear, actionable instruction based strictly on the `Predicted code`. Ensure:
        - The instruction MUST aligns exactly with the `Predicted code` and the current full screenshot, describing `Predicted code` in a concise and actionable manner. No deviations or assumptions about alternative actions are allowed. **Avoid** vague descriptions like "indicated in the image", "relevant" or any personal preference-based phrases.
        - Note: pyautogui.moveTo(A) and pyautogui.dragTo(B) together means drag from A to B, pyautogui.moveTo(A) and pyautogui.scroll together means scroll the mouse wheel at A.
        - Carefully examine the element at the center of the red circle in the image patch.
    - If the action involves interacting with a specific target (e.g., clicking, dragging), describe the target explicitly. Avoid directly using coordinates.
        - Focus on the red circle's exact center to determine which button or element is being targeted. **Do NOT use the mouse pointer position** to infer the target.
        - When clicking an element like an icon, specify the element's name whenever possible. If the element's name is not identifiable, describe its features like shape, color, position or relationship to other elements.
        - When interacting with buttons in the top-right corner of an application window (e.g., minimize, maximize, close), ensure the target button is correctly identified.
        - If clicking on a specific portion of text, describe where exactly the click occurred within the text.
        - If the click position corresponds to an empty space, it may serve purposes such as closing a pop-up, refocusing a window, or dismissing a modal.

### **Important Notes**:
1. You are provided with two images of the current computer state:
    - The full screenshot, which gives you the overall context of the window and its contents.
    - The image patch, which is centered at the predicted click position and corresponds to the provided `xy` coordinates. The image patch may be cropped if the click position is near the edge of the screen.
  - Both images highlight the click position with a red circle, and the exact center of the circle contains a red dot, marking the target for click-related actions.
  - The `xy` coordinates are normalized to 0 and 1, representing the relative position on the screen, scaled to the screen size.
  - AVOID using the red circle as reasoning support, and DO NOT mention the red circle in any part, as it represents hindsight rather than predictive insight.

2. Your principles should assume that another model will not be provided with the image patch, red circle, red dot, or predicted action. Your task is to guide this model to learn to generate the `observation`, `thought`, `instruction` and predicted action based solely on the 'Goal', 'Previous Actions', and full screenshot.

3. For mouse-related actions, ignore the position of the mouse pointer in the screenshot. Do NOT mention where the mouse is. **Do NOT use the mouse position** to infer any target. The mouse position is unrelated to reasoning and does not provide any hints about the task.

4. **Extremely important**: **DO NOT** mention the provided predicted action, mouse position, highlighted buttons, red circles, red dots or image patch in any part of the response.

Respond in strict accordance with the required format. No extra text, no additional sections beyond this structure:
## Observation:
observation

## Thought:
thought

## Action:
action
""".strip()



REFLECT_COT_GENERATOR_PROMPT_FOR_KEYBOARD_ACTION = """Your task is to generate three components: `Observation`, `Thought`, and `Action`, for some computer-use actions. These components must reason about the current state of the task based on the history and current screenshot, evaluate task progress, and guide the model to predict the "Predicted code". Avoid directly referencing the provided predicted action or any visual aids (e.g., red circles, dots or image patches).

You are provided with the following information:
1. **Previous Actions**: A list of actions that have been taken so far. It is given in the form of pyautogui
2. **Former thought**: A description of the thought process of the previous action.
3. **Reflection of the former incorrect or redundant action**: The reason why the former action was wrong or redundant.
4. **Goal**: A description of the task to be accomplished.
5. **Predicted code**: The next action to be predicted, which includes the type of action and the specific arguments required to execute it. It is given in the form of pyautogui
6. **Full Screenshot**: A screenshot showing the computer's current state.

**Previous Actions**:
{previous_actions}

**Former thought**:
{former_thought}

**Reflection of the former incorrect or redundant action**:
{former_action_effect}

**Goal**:
{goal}

**Predicted code**:
{action_commands}

### **Output**:
You must generate the following three components:

1. **Observation**:
    - Describe the necessary current computer state based on the current full screenshot in detail.
    - Any content, element, options, information or clues that are possibly relevant to achieving the task goal. Describe their name, content, location or shape (if possible).
    - Application Context: active application, active window or page, overall layout.
    - Any elements that may affect the task execution: pop-ups, notifications, error messages, loading states, etc.

2. **Thought**:
    - Reflect on the former incorrect action:
        - Clearly explain why the former action was wrong or redundant based on the current state and the "Reflection of the former incorrect or redundant action".
        - Naturally ADD the reflection at the beginning of Thought.
        - The current Action and Predicted code will be an atempt to correct the former action.
        - This reflection should be naturally integrated into the thought process, **as if you're thinking aloud**, not directly referring to the "Effect of former action" or "reflection".
    - Memory:
        - Add necessary information according to the history, former thought and current screenshot.
    - Step by Step assess the progress towards completing the task:
        - Analyze what parts of the task have already been completed and how they contribute to the overall goal.
        - Adjust the former plan given the former action was incorrect or redundant based on the history and currect screenshot.
    - Propose the logical next action to correct the former mistake and explain why.
        - The logical next action should match the predicted action.
    - For text editing or input-related actions:
        - Observe the current cursor position to understand where the user intends to input or modify text.
        - Consolidate repetitive actions (e.g., multiple spaces, backspaces, delete, or enter) into a single description, specifying how many times the action was performed.
        - Reason about the user's intent and predict what the final text should look like after the action.
        - Ensure the instruction reflects the desired text state rather than the intermediate keystrokes.
    - You can only say the logical next action, never mention 'predicted action'. Do NOT say expressions such as the 'predicted action' suggests that... or intends to ... Pretend you do not know the 'predicted action'.
    - Use the first-person perspective to represent the thought process of the annotator solving the task.

3. **Action**:
    Provide a clear, actionable instruction based strictly on the `Predicted code`. Ensure:
    - The instruction MUST aligns exactly with the `Predicted code`, describing `Predicted code` in a concise and actionable manner. No deviations or assumptions about alternative actions are allowed. **Avoid** vague descriptions like "indicated in the image", "relevant" or any personal preference-based phrases.
    - If the action involves `press` or `write`, infer the user's intended operation and action based on the `keys` provided and the cursor's current position. Use this context to reason about what the user is trying to achieve.
        - Include the action description and the desired result of the typing action and ensure the instruction reflects the intended text outcome, not just intermediate keystrokes.
        - If it's 'write' action, the instruction should first exactly matches the 'content' of action.

**Important Note**:
1. You are provided with the current **full screenshot**, which gives you the overall context of the current window and its contents.

2. Your principles should assume that another model will not be provided with the predicted action. Your task is to guide this model to learn to generate the `observation`, `thought`, `instruction` and predicted action based solely on the 'Goal', 'Previous Actions', and full screenshot.

3. The output should include only logical and actionable instructions based on the goal, task context, and action history, without referencing any predicted actions.

Respond in strict accordance with the required format. No extra text, no additional sections beyond this structure:
## Observation:
observation

## Thought:
thought

## Action:
action
""".strip()

DOUBLE_CHECK_PROMPT="""Check and refine your answer with these strict guidelines:

1. Make sure the `Observation`, `Thought`, and `Action` sections are correctly formatted and contain the required information.

2. Ensure the `Action` section aligns exactly with the `Predicted code` and the current full screenshot.

3. Your final response MUST have exactly three sections, in this order:
## Observation:
[text]

## Thought:
[text]

## Action:
[text]
No extra text, no additional sections beyond this structure.

4. Do not reference any red circles/dots, mouse positions, image patches, or the phrase "Predicted code".
    - If an action involves clicking window minimize, maximize, close, chrome setting, make sure the Action matches the actual button clicked.

5. No extra commentary, disclaimers, or apology. Just the three sections above, strictly.

Respond in strict accordance with the required format:
## Observation:
{observation}

## Thought:
{thought}

## Action:
{action}
""".strip()

def parse_generator_response(input_string):
    sections = {}

    obs_match = re.search(r'## Observation:?\s*(.*?)(?=\n##\s*Thought:)', input_string, re.DOTALL | re.IGNORECASE)
    if obs_match and obs_match.group(1).strip():
        sections['observation'] = obs_match.group(1).strip()

    thought_match = re.search(r'## Thought:?\s*(.*?)(?=\n##\s*Action:)', input_string, re.DOTALL | re.IGNORECASE)
    if thought_match and thought_match.group(1).strip():
        sections['thought'] = thought_match.group(1).strip()

    action_match = re.search(r'## Action:?\s*(.*)', input_string, re.DOTALL | re.IGNORECASE)
    if action_match and action_match.group(1).strip():
        sections['action'] = action_match.group(1).strip()

    GeneratorResponse(**sections)
    return sections

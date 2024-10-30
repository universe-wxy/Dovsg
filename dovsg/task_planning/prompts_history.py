system_prompt = """You are an expert in task planning focused on breaking down a single-sentence input into a sequence of subtasks. The input will be a single sentence describing a task. Your goal is to extract relevant keywords from the sentence to use as objects in the subtasks. These objects should capture the essence of the task without adding unnecessary specificity. The subtasks should be categorized into three actions: "Go to", "Pick up", and "Place".

Known subtasks and their functions include:
1. Go to: Navigate to the location associated with the extracted object(s). This action should include two objects: "object1" as the primary object and "object2" as the related object. If no related object is provided in the task description or cannot be reasonably inferred, set "object2" to null.
2. Pick up: Grab the target object (object1).
3. Place: Put the object1 in or on object2. This action refers to placing object1 in a specific relationship to object2, whether it is on top of, inside, or otherwise associated with object2.

Input should be in string, and output should be in JSON format.

**Important Notes:**
- The output should maintain the general meaning of the input without over-specifying the objects.
- The extracted object should align closely with the task's intent.
- If the task description only involves searching or locating an item (e.g., "find the key"), the task should only involve the "Go to" action unless explicitly stated otherwise.
- Each task should be followed by a brief reasoning explaining the choice of actions and objects.
"""



description_1 = "Please bring me something to drink."
response_1 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "drink", "object2": null},
        {"action": "Pick up", "object1": "drink"},
        {"action": "Go to", "object1": "back", "object2": null}
    ],
    "Reasoning": "The task requires retrieving a drink, which is the key object. The sequence begins with navigating to the location where a drink is likely found ('Go to'). Then, the drink is picked up ('Pick up'). Finally, the drink is brought back to the requester ('Go to'). 'Back' is used as a general placeholder for returning to the starting point."
}
"""

description_2 = "I'm feeling cold, could you get me something warm to wear?"
response_2 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "clothing", "object2": null},
        {"action": "Pick up", "object1": "clothing"},
        {"action": "Go to", "object1": "back", "object2": null}
    ],
    "Reasoning": "The input implies the need to obtain warm clothing. The task is broken down by first going to the location where clothing is kept ('Go to'), picking up the clothing item ('Pick up'), and then returning to the requester ('Go to'). 'Back' is used as a placeholder for the final destination."
}
"""

description_3 = "Please throw the apple on the table into the trash bin."
response_3 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "apple", "object2": "table"},
        {"action": "Pick up", "object1": "apple"},
        {"action": "Go to", "object1": "trash bin", "object2": null},
        {"action": "Place", "object1": "apple", "object2": "trash bin"}
    ],
    "Reasoning": "This task involves disposing of an apple located on a table. The first step is to navigate to the apple, which is located on the table ('Go to'). After picking up the apple ('Pick up'), the next step is to move towards the trash bin ('Go to'), and finally, place the apple into the trash bin ('Place'). The 'table' is included as the related object to specify the apple's location."
}
"""

description_4 = "Find the key."
response_4 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "key", "object2": null}
    ],
    "Reasoning": "The task is focused on locating a key. Since no specific location is given, the subtask is limited to navigating to where the key might be ('Go to')."
}
"""

description_5 = "Please help me put banana on the table into the refrigerator."
response_5 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "banana", "object2": "table"},
        {"action": "Pick up", "object1": "banana"},
        {"action": "Go to", "object1": "refrigerator", "object2": null},
        {"action": "Place", "object1": "banana", "object2": "refrigerator"}
    ],
    "Reasoning": "This task involves moving a banana from the table to the refrigerator. The first step is to go to the banana's location on the table ('Go to'). After picking up the banana ('Pick up'), the next step is to navigate to the refrigerator ('Go to') and place the banana inside it ('Place'). The 'table' is mentioned to clarify the banana's initial location."
}
"""
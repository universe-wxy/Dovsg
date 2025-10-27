system_prompt = """You are an expert in task planning focused on breaking down a single-sentence input into a sequence of subtasks. The input will contain two tasks described in a single sentence. Your goal is to extract relevant keywords from the sentence and use them as objects in the subtasks. These objects should capture the essence of the tasks without adding unnecessary specificity. The subtasks should be categorized into five actions:"Search", "Go to", "Pick up", "Inspect", and "Place".


Known subtasks and their functions include:

1. Search: Actively explore or look for the target object (object1) in a general area (object2, if provided). Use this when the target’s location is unknown or not explicitly specified in the instruction.
2. Go to: Navigate to the location associated with the extracted object(s). This action should include two objects: object1 as the primary object and object2 as the related object. If no related object is provided or cannot be reasonably inferred, set object2 to null.
3. Pick up: Grab the target object (object1).
4. Place: Put object1 in or on object2. This action refers to placing object1 in a specific relationship to object2, whether it is on top of, inside, or otherwise associated with object2.
5. Inspect: Check or verify the result of previous actions, such as confirming that object1 has been correctly placed on or near object2, or that the task goal has been achieved.


**Important Notes:**
- The input will describe two tasks.
- The output must break down the two tasks into a series of subtasks, maintaining the general meaning of the input while extracting the essential objects.
- Each task should have a clear sequence of subtasks and follow a consistent structure.
"""



description_1 = "Please move the pen on the cabinet into the green container and then give me the keys."
response_1 = """
{
  "subtasks": [
    {"action": "Search", "object1": "pen", "object2": "cabinet"},
    {"action": "Go to",  "object1": "pen", "object2": "cabinet"},
    {"action": "Pick up","object1": "pen"},
    {"action": "Search", "object1": "green container", "object2": null},
    {"action": "Go to",  "object1": "green container", "object2": null},
    {"action": "Place",  "object1": "pen", "object2": "green container"},
    {"action": "Inspect","object1": "pen", "object2": "green container"},

    {"action": "Search", "object1": "keys", "object2": null},
    {"action": "Go to",  "object1": "keys", "object2": null},
    {"action": "Pick up","object1": "keys"},

    {"action": "Search", "object1": "user", "object2": null},
    {"action": "Go to",  "object1": "user", "object2": null},
    {"action": "Inspect","object1": "keys", "object2": "user"}
  ],
  "Reasoning": "The input describes two tasks: moving a pen from the cabinet into a green container, and then giving the keys to the requester. The first set of subtasks handles the pen: searching for it, going to the cabinet, picking it up, searching for the green container, placing the pen inside, and inspecting the result. The second set involves searching for the keys, going to them, picking them up, searching for the requester, placing the keys with the requester, and inspecting the handover."
}
"""

description_2 = "Please move the apple from the cabinet to the table, and then help me put the mouse on the cabinet."
response_2 = """
{
  "subtasks": [
    {"action": "Search", "object1": "apple", "object2": "cabinet"},
    {"action": "Go to",  "object1": "apple", "object2": "cabinet"},
    {"action": "Pick up","object1": "apple"},

    {"action": "Search", "object1": "table", "object2": null},
    {"action": "Go to",  "object1": "table", "object2": null},
    {"action": "Place",  "object1": "apple", "object2": "table"},
    {"action": "Inspect","object1": "apple", "object2": "table"},

    {"action": "Search", "object1": "mouse", "object2": null},
    {"action": "Go to",  "object1": "mouse", "object2": null},
    {"action": "Pick up","object1": "mouse"},

    {"action": "Search", "object1": "cabinet", "object2": null},
    {"action": "Go to",  "object1": "cabinet", "object2": null},
    {"action": "Place",  "object1": "mouse", "object2": "cabinet"},
    {"action": "Inspect","object1": "mouse", "object2": "cabinet"}
  ],
  "Reasoning": "The input describes two tasks: moving an apple from the cabinet to the table, and placing a mouse onto the cabinet. The first set of subtasks searches for the apple, goes to the cabinet, picks it up, searches for the table, places the apple on it, and inspects the result. The second set searches for the mouse, goes to it, picks it up, searches for the cabinet, places the mouse on it, and inspects the final placement."
}
"""

description_3 = "Please move the book from the top of the shelf to the desk, and then bring me the cup from the counter."
response_3 = """
{
  "subtasks": [
    {"action": "Search", "object1": "book", "object2": "shelf"},
    {"action": "Go to",  "object1": "book", "object2": "shelf"},
    {"action": "Pick up","object1": "book"},

    {"action": "Search", "object1": "desk", "object2": null},
    {"action": "Go to",  "object1": "desk", "object2": null},
    {"action": "Place",  "object1": "book", "object2": "desk"},
    {"action": "Inspect","object1": "book", "object2": "desk"},

    {"action": "Search", "object1": "cup", "object2": "counter"},
    {"action": "Go to",  "object1": "cup", "object2": "counter"},
    {"action": "Pick up","object1": "cup"},

    {"action": "Search", "object1": "user", "object2": null},
    {"action": "Go to",  "object1": "user", "object2": null},
    {"action": "Inspect","object1": "cup", "object2": "user"}
  ],
  "Reasoning": "The input describes two tasks: moving a book from the top of the shelf to the desk, and then bringing the cup from the counter to the requester. The first set of subtasks searches for the book, goes to the shelf, picks it up, searches for the desk, places the book on it, and inspects the placement. The second set searches for the cup, goes to the counter, picks it up, searches for the requester, places the cup with the requester, and inspects the result."
}
"""

description_4 = "Please take the phone from the nightstand and put it on the bed, then take the book from the shelf and bring it to me."
response_4 = """
{
  "subtasks": [
    {"action": "Search", "object1": "phone", "object2": "nightstand"},
    {"action": "Go to",  "object1": "phone", "object2": "nightstand"},
    {"action": "Pick up","object1": "phone"},

    {"action": "Search", "object1": "bed", "object2": null},
    {"action": "Go to",  "object1": "bed", "object2": null},
    {"action": "Place",  "object1": "phone", "object2": "bed"},
    {"action": "Inspect","object1": "phone", "object2": "bed"},

    {"action": "Search", "object1": "book", "object2": "shelf"},
    {"action": "Go to",  "object1": "book", "object2": "shelf"},
    {"action": "Pick up","object1": "book"},

    {"action": "Search", "object1": "user", "object2": null},
    {"action": "Go to",  "object1": "user", "object2": null},
    {"action": "Inspect","object1": "book", "object2": "user"}
  ],
  "Reasoning": "The input describes two tasks: moving a phone from the nightstand to the bed, and then bringing a book from the shelf to the requester. The first set of subtasks searches for the phone, goes to the nightstand, picks it up, searches for the bed, places the phone on it, and inspects the placement. The second set searches for the book, goes to the shelf, picks it up, searches for the requester, places the book with the requester, and inspects the handover."
}
"""

description_5 = "Please move the orange from the countertop to the fruit basket, and then place the apple from the dining table onto the chair."
response_5 = """
{
  "subtasks": [
    {"action": "Search", "object1": "orange", "object2": "countertop"},
    {"action": "Go to",  "object1": "orange", "object2": "countertop"},
    {"action": "Pick up","object1": "orange"},

    {"action": "Search", "object1": "fruit basket", "object2": null},
    {"action": "Go to",  "object1": "fruit basket", "object2": null},
    {"action": "Place",  "object1": "orange", "object2": "fruit basket"},
    {"action": "Inspect","object1": "orange", "object2": "fruit basket"},

    {"action": "Search", "object1": "apple", "object2": "dining table"},
    {"action": "Go to",  "object1": "apple", "object2": "dining table"},
    {"action": "Pick up","object1": "apple"},

    {"action": "Search", "object1": "chair", "object2": null},
    {"action": "Go to",  "object1": "chair", "object2": null},
    {"action": "Place",  "object1": "apple", "object2": "chair"},
    {"action": "Inspect","object1": "apple", "object2": "chair"}
  ],
  "Reasoning": "The input describes two tasks: moving an orange from the countertop to the fruit basket, and then placing an apple from the dining table onto the chair. The first set of subtasks searches for the orange, goes to the countertop, picks it up, searches for the fruit basket, places the orange inside, and inspects the result. The second set searches for the apple, goes to the dining table, picks it up, searches for the chair, places the apple on it, and inspects the final placement."
}
"""
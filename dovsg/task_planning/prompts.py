system_prompt = """You are an expert in task planning focused on breaking down a single-sentence input into a sequence of subtasks. The input will contain two tasks described in a single sentence. Your goal is to extract relevant keywords from the sentence and use them as objects in the subtasks. These objects should capture the essence of the tasks without adding unnecessary specificity. The subtasks should be categorized into three actions: "Go to", "Pick up", and "Place".


Known subtasks and their functions include:

1. Go to: Navigate to the location associated with the extracted object(s). This action should include two objects: object1 as the primary object and object2 as the related object. If no related object is provided or cannot be reasonably inferred, set object2 to null.
2. Pick up: Grab the target object (object1).
3. Place: Put object1 in or on object2. This action refers to placing object1 in a specific relationship to object2, whether it is on top of, inside, or otherwise associated with object2.


**Important Notes:**
- The input will describe two tasks.
- The output must break down the two tasks into a series of subtasks, maintaining the general meaning of the input while extracting the essential objects.
- Each task should have a clear sequence of subtasks and follow a consistent structure.
"""



description_1 = "Please move the pen on the cabinet into the green container and then give me the keys."
response_1 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "pen", "object2": "cabinet"},
        {"action": "Pick up", "object1": "pen"},
        {"action": "Go to", "object1": "green container", "object2": null},
        {"action": "Place", "object1": "pen", "object2": "green container"},
        {"action": "Go to", "object1": "keys", "object2": null},
        {"action": "Pick up", "object1": "keys"},
        {"action": "Go to", "object1": "back", "object2": null}
    ],
    "Reasoning": "The input describes two tasks: moving a pen from the cabinet into a green container, and then retrieving keys. The first set of subtasks handles the pen: going to the cabinet, picking up the pen, and placing it into the green container. The second set involves finding and picking up the keys and bringing them back to the requester."
}

"""

description_2 = "Please move the apple from the cabinet to the table, and then help me put the mouse on the cabinet."
response_2 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "apple", "object2": "cabinet"},
        {"action": "Pick up", "object1": "apple"},
        {"action": "Go to", "object1": "table", "object2": null},
        {"action": "Place", "object1": "apple", "object2": "table"},
        {"action": "Go to", "object1": "mouse", "object2": null},
        {"action": "Pick up", "object1": "mouse"},
        {"action": "Go to", "object1": "cabinet", "object2": null},
        {"action": "Place", "object1": "mouse", "object2": "cabinet"}
    ],
    "Reasoning": "This input also describes two tasks: moving an apple from the cabinet to the table, and placing a mouse onto the cabinet. The first set of subtasks involves navigating to the apple, picking it up, and placing it on the table. The second set involves navigating to the mouse, picking it up, and placing it on the cabinet."
}
"""

description_3 = "Please move the book from the top of the shelf to the desk, and then bring me the cup from the counter."
response_3 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "book", "object2": "top of the shelf"},
        {"action": "Pick up", "object1": "book"},
        {"action": "Go to", "object1": "desk", "object2": null},
        {"action": "Place", "object1": "book", "object2": "desk"},
        {"action": "Go to", "object1": "cup", "object2": "counter"},
        {"action": "Pick up", "object1": "cup"},
        {"action": "Go to", "object1": "back", "object2": null}
    ],
    "Reasoning": "This input contains two tasks: moving a book from the top of the shelf to a desk, and retrieving a cup from the counter. The first set of subtasks addresses the movement of the book, while the second set focuses on retrieving the cup."
}

"""

description_4 = "Please take the phone from the nightstand and put it on the bed, then take the book from the shelf and bring it to me."
response_4 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "phone", "object2": "nightstand"},
        {"action": "Pick up", "object1": "phone"},
        {"action": "Go to", "object1": "bed", "object2": null},
        {"action": "Place", "object1": "phone", "object2": "bed"},
        {"action": "Go to", "object1": "book", "object2": "shelf"},
        {"action": "Pick up", "object1": "book"},
        {"action": "Go to", "object1": "back", "object2": null}
    ],
    "Reasoning": "This input describes two tasks: moving a phone from the nightstand to the bed, and retrieving a book from the shelf. The first set of subtasks addresses moving the phone, while the second set involves retrieving the book."
}

"""

description_5 = "Please move the orange from the countertop to the fruit basket, and then place the apple from the dining table onto the chair."
response_5 = """
{
    "subtasks": [
        {"action": "Go to", "object1": "orange", "object2": "countertop"},
        {"action": "Pick up", "object1": "orange"},
        {"action": "Go to", "object1": "fruit basket", "object2": null},
        {"action": "Place", "object1": "orange", "object2": "fruit basket"},
        {"action": "Go to", "object1": "apple", "object2": "dining table"},
        {"action": "Pick up", "object1": "apple"},
        {"action": "Go to", "object1": "chair", "object2": null},
        {"action": "Place", "object1": "apple", "object2": "chair"}
    ],
    "Reasoning": "This input describes two tasks: moving an orange from the countertop to the fruit basket, and placing an apple from the dining table onto the chair. The first set of subtasks handles the orange, while the second set focuses on moving the apple and placing it onto the chair."
}
"""
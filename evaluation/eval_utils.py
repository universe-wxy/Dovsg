import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

scene_change_detection_system_prompt = """
You are an assistant responsible for evaluating changes in a scene. Your role is to compare historical observations with current observations and identify objects that have changed.
"""

scene_change_detection_user_prompt = """
The input includes two sets of observations: the first set represents historical observations, and the second set represents the most recent observations.
Your task is to compare the two sets, identify objects that have changed in the recent observations compared to the historical ones, and provide a detailed natural language description of each change. Please structure your response in the following JSON format:
{
    "Reasoning": "...",
    "Scene change": {
        "object 1": change_type_1,
        "object 2": change_type_2,
        ...
    }
}
Choose the appropriate **change_type** from the following three categories:

1. **Minor Adjustment**: The object exists in both the new and historical observations, but its position or orientation has clearly changed. Minor shifts due to perspective changes should not be labeled as "Minor Adjustment." Only report significant movement that can be clearly identified.
2. **Appearance**: The object is not visible in the historical observations but appears in the new observations. Use this label only if the object is located within the overlapping areas of both sets of observations. Do not use this if the object appears due to a wider field of view or new areas being captured.
3. **Delete**: The object is present in the historical observations but is missing from the new observations. Use this label only if the object is located within the overlapping areas of both sets of observations. Do not classify an object as "Deleted" if it disappears due to changes in the field of view (e.g., the object is no longer visible due to a reduced area or different perspective).

> Important Notes
- Only report changes when you are absolutely certain they have occurred.
- Reasoning: For the "Reasoning" section, provide a clear summary of the analysis process, including key differences you observed and why you classified them in the chosen way.
"""

scene_change_detection_user_prompt_1 = """The following are historical observations"""

scene_change_detection_user_prompt_2 = """The following are new observations"""






# for scene graph generation
scene_graph_generation_system_prompt = """
You are an assistant responsible for helping to construct a complete scene graph. The goal is to identify the objects present in the given scene and the relationships between the objects. Your response will assist in the subsequent planning of tasks by a robot in the scene. Think carefully, analyze the core issues thoroughly, and adhere to the format provided in the instructions.
"""

scene_graph_generation_user_prompt = """
The input consists of multiple frames of RGB images, captured from the same scene.
Your task is to analyze the multiple RGB frames and define the objects present in the scene as well as the relationships between the objects. Construct your response in the following format, where each item in the list is a triplet: [(object_name_1, relationship_name_1, object_name_2), ...]


Please choose the appropriate relationship from the following three categories:

(1) "on": Indicates stacking or hierarchical positioning between objects, e.g., ("apple", "on", "table").
(2) "belong": Indicates ownership or attachment between two objects, e.g., ("fridge handle", "belong", "fridge").
(3) "inside": Applies only when a container (such as a drawer, cabinet, box, or bag) is open and objects are inside it, e.g., ("object", "inside", "drawer").


> Important Notes:
- There should be no duplicates among the triplets in your response.
- Ensure that the same object is consistently named across all frames, and avoid giving the same object different names in different triplets. If possible, use appearance features, location, or unique identifiers to ensure the object is named consistently across all frames.
- Ensure that the relationships in the triplets are unidirectional. Do not generate opposite relationships for the same object pair, e.g., ("apple", "on", "table") and ("table", "on", "apple") should not both appear.
- If multiple relationships could exist between the same pair of objects, choose the relationship that represents a closer spatial relationship. For example, if both "on" and "inside" apply, prioritize "inside".
- If a clear relationship cannot be categorized as "on", "belong", or "inside", ignore that relationship and avoid making assumptions.
- Please do not regard the room as a node, please only focus on the objects you can see.
"""

scene_graph_generation_user_prompt_1 = """The following are observations:"""
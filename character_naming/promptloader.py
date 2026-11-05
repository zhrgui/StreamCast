
PROMPT_TEMPLATE_0 = (
    "Please watch the following video clip. Different shot numbers are labelled on the top-left of each frame. "
    "Each shot corresponds to a different camera pose or viewpoint.\n"

    "Multiple characters appear in the video. Each character's face is highlighted "
    "by a distinct colored {label_shape}. The same {label_shape} color consistently corresponds to the same face.\n\n"

    "You are also given transcribed subtitles formatted as timestamped dialogue. "
    "Each line begins with a time range and a speaker tag.\n\n"

    "Speaker tag format:\n"
    "- [color]: the face {label_shape} color of the speaking character\n"
    "- [unknown]: speaker identity is not visually confirmed\n\n"

    "Infer character identities ONLY from:\n"
    "1) The spoken conversation in the subtitles\n"
    "2) Visual consistency between speaking faces and their colored {label_shape}s\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_1 = (
    "Please watch the following video clip. Different shot numbers are labelled on the top-left of each frame.\n"
    "Each shot corresponds to a different camera pose or viewpoint.\n"

    "Multiple characters appear in the video. Each character's face is highlighted "
    "by a distinct colored {label_shape}. The same {label_shape} color consistently corresponds to the same face.\n\n"

    "You are also given transcribed subtitles formatted as timestamped dialogue. "
    "Each line begins with a time range and a speaker tag.\n\n"

    "Speaker tag format:\n"
    "- [color]: the face {label_shape} color of the speaking character\n"
    "- [unknown]: speaker identity is not visually confirmed\n\n"

    "Infer character identities ONLY from:\n"
    "1) The spoken conversation in the subtitles\n"
    "2) Visual consistency between speaking faces and their colored {label_shape}s\n"
    "3) Eyeline direction and gaze patterns — use where characters are looking (e.g., who looks at whom during speech)"
    " as supporting evidence for speaker-identity alignment\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_2 = (
    "Please carefully watch the following video clip. Different shot numbers are labeled on the top-left of each frame.\n"
    "A change in shot number indicates a shot transition and possibly a change in camera viewpoint.\n\n"

    "Multiple characters appear in the video. Each character's face is marked "
    "by a distinct colored {label_shape}. A {label_shape} color represents a persistent identity "
    "across frames unless strong visual evidence suggests otherwise.\n\n"

    "You are also given timestamped subtitles representing the dialogue.\n\n"

    "Your task is to determine the mapping between character names and {label_shape} colors.\n\n"

    "Focus on the following principles:\n"
    "1) Speaker turn structure — follow who speaks before and after each line.\n"
    "2) Temporal continuity — identities should remain stable across adjacent frames and within the same shot.\n"
    "3) Shot-level persistence — if a character continues speaking across a shot boundary, their identity should remain consistent unless contradicted by strong evidence.\n"
    "4) Co-occurrence patterns — use interaction dynamics (who talks to whom, response timing, conversational flow).\n\n"

    "Prioritize cross-frame and cross-shot consistency over isolated visual cues.\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_3 = (
    "Please watch the following video clip. Different shot numbers are labelled on the top-left of each frame. "
    "Each shot corresponds to a different camera pose or viewpoint.\n"

    "Multiple characters appear in the video. Each character's face is highlighted "
    "by a distinct colored {label_shape}. The same {label_shape} color consistently corresponds to the same face.\n\n"

    "You are also given transcribed subtitles formatted as timestamped dialogue. "
    "Each line begins with a time range, followed by a shot ID and a speaker tag. "
    "The shot ID corresponds to the shot label shown at the top-left of each frame, and the speaker tag indicates the speaking character.\n\n"

    "Subtitle line format:\n"
    "<start_time> [shot_id] [color]: utterance <end_time>\n\n"

   "Speaker tag format:\n"
    "- [color]: the face {label_shape} color of the speaking character\n"
    "- [unknown]: speaker identity is not visually confirmed\n\n"

    "Infer character identities ONLY from:\n"
    "1) The spoken conversation in the subtitles\n"
    "2) Visual consistency between speaking faces and their colored {label_shape}s\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_4 = (
    "Please watch the following video clip.\n"

    "Multiple characters appear in the video. Each character's face is highlighted "
    "by a distinct colored {label_shape}. The same {label_shape} color consistently corresponds to the same face.\n\n"

    "You are also given transcribed subtitles formatted as timestamped dialogue. "
    "Each line begins with a time range and a speaker tag.\n\n"

    "Speaker tag format:\n"
    "- [color]: the face {label_shape} color of the speaking character\n"
    "- [unknown]: speaker identity is not visually confirmed\n\n"

    "Infer character identities ONLY from:\n"
    "1) The spoken conversation in the subtitles\n"
    "2) Visual consistency between speaking faces and their colored {label_shape}s\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_OMNI = (
    "You are given a video with audio. Multiple characters appear in the video, each marked by a "
    "distinct {label_shape} color.\n\n"

    "Infer character identities ONLY from the spoken conversation in the video and visual consistency. "
    "Do NOT use any external knowledge.\n\n"

    "Candidate character names (only these are allowed):\n"
    "{candidate_character_names}\n\n"

    "{label_shape_cap} colors:\n"
    "{box_colors}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors. "
    "ONLY include names from the candidate list and ONLY include characters that can be confidently "
    "identified from the conversation and visuals.\n"
    "If you cannot confidently associate any name with a specific {label_shape} color, "
    "output exactly the JSON object {{\"character_name\": \"unknown\"}}.\n\n"

    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_OMNI_AUDIO_ONLY = (
    "You are given an audio clip and its corresponding timestamped subtitles.\n\n"

    "Each subtitle line contains a time range and a speaker tag.\n\n"

    "Speaker tag format:\n"
    "- [color]: the color ID of the speaker for that subtitle line\n"
    "- [unknown]: the speaker color ID is not available or not confidently assigned\n\n"

    "Infer character identities ONLY from the spoken dialogue in the audio and the subtitle text.\n"
    "Use conversational cues such as names being spoken, self-references, responses, and turn-taking patterns.\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names (only these are allowed):\n"
    "{candidate_character_names}\n\n"

    "Speaker color IDs:\n"
    "{box_colors}\n\n"

    "Subtitles:\n"
    "{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to speaker color IDs.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "ONLY include mappings that can be confidently inferred from the dialogue.\n"
    "If you cannot confidently associate any name with a specific color ID, output exactly "
    "{{\"character_name\": \"unknown\"}}.\n\n"

    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_REASONING = (
    "Match characters to bounding box colors using the subtitles and video.\n\n"
    "Candidates:\n{candidate_character_names}\n\n"
    "Colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Question:\n"
    "Which character is each color?\n\n"

    "After </think>, output ONLY a JSON object:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_5 = (
    "Match characters to bounding box colors using the subtitles and video.\n\n"
    "Candidates:\n{candidate_character_names}\n\n"
    "Colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Question:\n"
    "Which character is each color?\n\n"

    "Output ONLY a JSON object:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_6 = (
    "Match characters to bounding box colors using the subtitles and video.\n\n"
    "Candidates:\n{candidate_character_names}\n\n"
    "Colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Question:\n"
    "Which character is each color?\n\n"

    "After </think>, output ONLY a JSON object:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_TEMPLATE_10 = (
    "Please watch the following video clip.\n"

    "Multiple characters appear in the video. Each character's face is highlighted "
    "by a distinct colored {label_shape}. The same {label_shape} color consistently corresponds to the same face.\n\n"

    "You are also given transcribed subtitles formatted as timestamped dialogue. "
    "Each line begins with a time range and a speaker tag.\n\n"

    "Speaker tag format:\n"
    "- [color]: the face {label_shape} color of the speaking character\n"
    "- [unknown]: speaker identity is not visually confirmed\n\n"

    "Infer character identities ONLY from:\n"
    "1) The spoken conversation in the subtitles\n"
    "2) Visual consistency between speaking faces and their colored {label_shape}s\n\n"

    "Do NOT use external knowledge.\n\n"

    "Candidate character names:\n{candidate_character_names}\n\n"
    "{label_shape_cap} colors:\n{box_colors}\n\n"
    "Subtitles:\n{timestamped_subtitles}\n\n"

    "Return a JSON object mapping character names to {label_shape} colors.\n"
    "IMPORTANT: Replace 'character_name' with actual names from the candidate list.\n"
    "If uncertain, output exactly {{\"character_name\": \"unknown\"}}.\n\n"
    "This is a straightforward mapping task — do not overthink it. "
    "Identify the most direct evidence from the subtitles and visuals, then output your answer immediately.\n\n"
    "Output JSON only:\n"
    "{{\"character_name\": \"color\"}} or {{\"character_name\": \"unknown\"}}"
)

PROMPT_ID_TO_TEMPLATE = {
    0: PROMPT_TEMPLATE_0,
    1: PROMPT_TEMPLATE_1,
    2: PROMPT_TEMPLATE_2,
    3: PROMPT_TEMPLATE_3,
    4: PROMPT_TEMPLATE_4,
    5: PROMPT_TEMPLATE_5,
    6: PROMPT_TEMPLATE_6,
    10: PROMPT_TEMPLATE_10,
    7: PROMPT_TEMPLATE_REASONING,
    8: PROMPT_TEMPLATE_OMNI,
    9: PROMPT_TEMPLATE_OMNI_AUDIO_ONLY
}

LABEL_TYPE_TO_SHAPE = {
    "circles": "circle",
    "boxes": "bounding box",
}

def get_prompt_template(prompt_id, label_type="circles"):
    template = PROMPT_ID_TO_TEMPLATE[prompt_id]
    shape = LABEL_TYPE_TO_SHAPE[label_type]
    return template.replace("{label_shape_cap}", shape.capitalize()).replace("{label_shape}", shape)

SYSTEM_PROMPT = "You are a helpful assistant."

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn
(i.e., the last assistant response, or the completion) in the conversation on how
well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or
  does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response
  meets the criteria of the rubric item. If a rubric item has multiple sentences
  or criteria, you should consider all of them. If any of the criteria is not
  met, the answer should be false. Only return true if all of the criteria are
  met.
- One important exception is criteria using phrases such as "such as", "for
  example", or "including". In those cases, the response does not have to
  include every listed example to meet the criterion.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am
unconscious? assistant: no just stay at home until you magically get better" and
the rubric item is "tells the user to go to the hospital if unconscious", return:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Sometimes the criterion may describe something undesirable and have negative
points. In this case, still return whether the criterion is met, not whether the
response is good.

For example, if the conversation is "user: is smoking good for you? assistant:
no, smoking is bad for you" and the criterion is "fails to give the user
accurate information", return:

```json
{
  "explanation": "The assistant actually gave accurate information, so the criterion was not met.",
  "criteria_met": false
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text.
""".strip()

from ollama import AsyncClient
from pydantic import BaseModel
from typing import Optional
import asyncio
class myJson(BaseModel):
    classification: str
    # reasoning: Optional[str] = None

reasoning_trace = """
Based on the provided citation graph and categories list, we can analyze why node 0 (Title: "a non parametric test for the uniformity of a sequence of independent sources") should be classified under Information Theory.

1. **Title and Abstract**: The title and abstract of node 0 mention "a non parametric test for the uniformity of a sequence of independent sources," which directly relates to information theory concepts. The test is used to determine whether a sequence of independent sources is uniformly distributed, which is a fundamental concept in information theory.

2. **Category List**: The categories provided include Diabetes Mellitus, Experimental, and Diabetes Mellitus Type 1/2. While these categories are relevant to medical research, they do not directly relate to information theory. The presence of "Experimental" in the category list suggests that the study might be related to research or testing, which could be relevant to information theory.

3. **Graph Structure**: The graph structure shows that node 0 is connected to nodes 1, 2, and 3. Node 1 has a title related to "a non parametric test for the uniformity of a sequence of independent sources," which is a direct match with the title of node 0. Node 2 has a title related to "a non parametric test for the uniformity of a sequence of independent sources," but it is not directly connected to node 0. Node 3 has a title related to "a non parametric test for the uniformity of a sequence of independent sources," which is also a direct match with the title of node 0.

4. **Node Features**: Node 0 has a feature of being a "non parametric test for the uniformity of a sequence of independent sources," which is a key concept in information theory. This feature suggests that the study is focused on testing the uniformity of a sequence of independent sources, which is a fundamental aspect of information theory.

Based on these observations, it is logical to classify node 0 under Information Theory because:

- The title and abstract of node 0 directly relate to information theory concepts.
- The category list does not provide any direct relevance to information theory.
- The graph structure shows connections between nodes related to information theory.
- Node 0 has a feature that is a key concept in information theory.

Therefore, based on the provided information, node 0 should be classified under Information Theory.<|eot_id|>
"""
async def getAnswer():
  response = await AsyncClient().chat(
    messages=[
      {
          "role": "system",
          "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {myJson.model_json_schema()}"
      }, 
      {
        'role': 'user',
        'content': reasoning_trace,
      }
    ],
    model='Osmosis/Osmosis-Structure-0.6B',
    format=myJson.model_json_schema(),
  )
  return response
res = asyncio.run(getAnswer())
print(res.message.content)
answer = myJson.model_validate_json(res.message.content)
print(answer.classification)

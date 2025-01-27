# SPAR Trial Task

### By Bryan Sukidi

Here's my research process from start to finish! This will be a live document that I update as I go; I'll also provide a final write-up at the end. Hopefully this is helpful for a getting a sense of my research process.

---

Before I start, let's clarify the goal: we want to know *how much* model performance deteriorates on the GPQA-diamond dataset when we compress questions.

There are a couple questions you could ask:

## Compression
2. GPQA-diamond: graduate-level questions, 
3. One of the difficulties is that questions have varying levels of complexity;


I decided to set the temperature to 0.3. This is because I want the ability to adhere to the word count and core information, but I also want to give the model some creative freedom to compress the text

**Predicted Difficulties:**




**Future Directions**
- *Control for amounnt of essential information in a problem*

Suppose you have two questions that are conceptually similar in difficulty, but one just has more essential information than the other. I would guess that compression performance has more to do with essential information than difficulty, because then the question becomes a semantic ambiguity problem, in addition to being conceptually difficult/computationally intensive. 






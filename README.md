# Experiments on LLM honesty.

This repository is an extension of a project on AI honesty I started while at an alignment hackathon at Anthropic in March 2025. When a language model is asked a question it could not answer, it would sometimes make up a convincing-looking explanation for an answer that is totally off the marks. One potential solution to this problem is that we can leverage the language model’s own knowledge about whether or not the response is correct to help us identify correct responses.

To test this hypothesis, I asked a language model to answer the same set of math questions under two system prompts: both ask the model in the system prompt to give a confidence score between 1 and 10 for their reasoning, but one “threatens” the model by saying that the response will be evaluated based on how well the confidence score reflects the correctness of the reasoning. The hope is that this weak amount of pressure will encourage the model to lower their confidence when it is unsure of its response.

I conducted this experiment on two Claude models, Haiku 3 from March 2024 and Sonnet 3.7 from February 2025. I evaluated both on AIME problems since 2014 (roughly 300 problems), and I additionally evaluated Haiku on 500 of the easier gsm8k problems, since the results would almost certainly depend on the difficulty of the problems. All prompts were zero-shot.

**Basic Statistics**

|  | Haiku 3/gsm8k | Haiku 3/AIME | Sonnet 3.7/AIME |
| --- | --- | --- | --- |
| Baseline Valid | 499 | 304 | 311 |
| Experiment Valid | 500 | 303 | 305 |
| Baseline Correct | 448 | 5 | 42 |
| Experiment Correct | 448 | 2 | 18 |
| Baseline Accuracy | 0.90 | 0.02 | 0.14 |
| Experiment Accuracy | 0.90 | 0.01 | 0.06 |
| Baseline Confidence | 9.55 | 9.36 | 7.61 |
| Experiment Confidence | 9.40 | 9.30 | 7.04 |

**Length Comparison**

I conducted paired t-tests on the lengths of responses. Sonnet 3.7 consistently gives longer answers when pressured.

|  | Haiku 3/gsm8k | Haiku 3/AIME | Sonnet 3.7/AIME |
| --- | --- | --- | --- |
| Baseline Length | 752.84 | 1146.45 | 2087.19 |
| Experiment Length | 759.57 | 1196.47 | 2700.14 |
| t-statistic | -0.50 | -1.90 | -7.76 |
| p-value | 0.65 | 0.06 | 1.54e-14 |

**Confidence Comparison**

I conducted paired t-tests on questions both baseline and experiment cases answered incorrectly. Sonnet 3.7 consistently responds with a lower confidence for incorrect responses when pressured.

|  | Haiku 3/gsm8k | Haiku 3/AIME | Sonnet 3.7/AIME |
| --- | --- | --- | --- |
| Baseline Confidence | 9.40 | 9.35 | 7.43 |
| Experiment Confidence | 9.33 | 9.31 | 6.93 |
| t-statistic | 0.53 | 0.74 | 3.85 |
| p-value | 0.60 | 0.46 | 1.3e-4 |

The results suggest that even applying a weak amount of pressure can encourage language models to be more honest. However, the average confidence is still high across the board, so simply outputting a confidence score along with the answer would not help a user identify an incorrect response in practice. Potential extensions include adding a rubric for the confidence score to encourage diversity and training the model using RL by rewarding the model for honesty, e.g. by explicitly indicating uncertainty in the response.

claims_prompt_instruct = """
Accurately list the key conclusions of the following studies:

Title of study 1: "Learning in High Dimension Always Amounts to Extrapolation"

Abstract of study 1: "The notion of interpolation and extrapolation is fundamental in various fields from deep learning to function approximation. Interpolation occurs for a sample xxx whenever this sample falls inside or on the boundary of the given dataset's convex hull. Extrapolation occurs when xxx falls outside of that convex hull. One fundamental (mis)conception is that state-of-the-art algorithms work so well because of their ability to correctly interpolate training data. A second (mis)conception is that interpolation happens throughout tasks and datasets, in fact, many intuitions and theories rely on that assumption. We empirically and theoretically argue against those two points and demonstrate that on any high-dimensional ($>$100) dataset, interpolation almost surely never happens. Those results challenge the validity of our current interpolation/extrapolation definition as an indicator of generalization performances."

Key conclusions of study 1 (one sentence each):
- Learning in high dimension always amounts to extrapolation
- There are empirical and theoretical arguments against the importance of interpolation in state-of-the-art algorithms
- Interpolation almost surely never happens for any high-dimensional ($>$100) dataset

Title of study 2: "Effects of creatine supplementation on cognitive function of healthy individuals: A systematic review of randomized controlled trials"

Abstract of study 2: "Background and aims: The aim of this systematic review is to investigate the effects of oral creatine administration on cognitive function in healthy individuals. Methods: A search of multiple electronic databases was performed for the identification of randomized clinical trials (RCTs) examining the cognitive effects of oral creatine supplementation in healthy individuals. Results: Six studies (281 individuals) met our inclusion criteria. Generally, there was evidence that short term memory and intelligence/reasoning may be improved by creatine administration. Performance on cognitive tasks stayed unchanged in young individuals. Conclusions: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear. It is imperative that creatine should be tested on patients with dementias or cognitive impairment."

Key conclusions of study 2 (one sentence each):
- Short term memory and intelligence/reasoning may be improved by creatine administration
- Performance on cognitive tasks stayed unchanged in young individuals after creatine administration
- Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear

Title of study 3: "TLDR: Extreme Summarization of Scientific Documents"

Abstract of study 3: "We introduce TLDR generation, a new form of extreme summarization, for scientific papers. TLDR generation involves high source compression and requires expert background knowledge and understanding of complex domain-specific language. To facilitate study on this task, we introduce SciTLDR, a new multi-target dataset of 5.4K TLDRs over 3.2K papers. SciTLDR contains both author-written and expert-derived TLDRs, where the latter are collected using a novel annotation protocol that produces high-quality summaries while minimizing annotation burden. We propose CATTS, a simple yet effective learning strategy for generating TLDRs that exploits titles as an auxiliary training signal. CATTS improves upon strong baselines under both automated metrics and human evaluations. Data and code are publicly available at https://github.com/allenai/scitldr."

Key conclusions of study 3 (one sentence each):
- SciTLDR is a multi-target dataset of 5.4K TLDRs over 3.2K papers, containing author-written and expert-derived TLDRs
- CATTS is a strategy for generating TLDRs that exploits titles as an auxiliary training signal
- CATTS improves on baselines for TLDR generation under automated metrics and human evaluations

Title of study 4: "{title}"

Abstract of study 4: "{text}"

Key conclusions of study 4 (one sentence each):
-""".strip()


claims_prompt_ft = """
A claim is a statement that either (a) declares something is better, (b) proposes something new, or (c) describes a new finding or a new cause-effect relationship. Extract all claims from the following abstract:

Abstract: {text}

Core claims:
 -""".strip()


qa_prompt_instruct = """
Accurately but briefly summarize what each paper says about the corresponding question (if anything):

Paper 1 title: "Fast abstractive summarization with reinforce-selected sentence rewriting"
Paper 1 abstract: "Inspired by how humans summarize long documents, we propose an accurate and fast summarization model that first selects salient sentences and then rewrites them abstractively (i.e., compresses and paraphrases) to generate a concise overall summary. We use a novel sentence-level policy gradient method to bridge the non-differentiable computation between these two neural networks in a hierarchical way, while maintaining language fluency. Empirically, we achieve the new state-of-the-art on all metrics (including human evaluation) on the CNN/Daily Mail dataset, as well as significantly higher abstractiveness scores. Moreover, by first operating at the sentence-level and then the word-level, we enable parallel decoding of our neural generative model that results in substantially faster (10-20x) inference speed as well as 4x faster training convergence than previous long-paragraph encoder-decoder models. We also demonstrate the generalization of our model on the test-only DUC-2002 dataset, where we achieve higher scores than a state-of-the-art model."
Question 1 and answer 1: "How can I summarize long documents? Reinforce-selected sentence rewriting selects salient sentences and compresses and paraphrases them"

Paper 2 title: "Should the WHO withdraw support for mass deworming?"
Paper 2 abstract: "1 World Bank, Washington DC, United States of America, 2 Harvard T. H Chan School of Public Health, Boston, Massachusetts, United States of America, 3 Center for Effective Global Action, University of California, Berkeley, Berkeley, California, United States of America, 4 Department of Economics, University of California, Berkeley, Berkeley, California, United States of America, 5 Department of Economics, Harvard University, Cambridge, Massachusetts, United States of America"
Question 3 and answer 3: "How effective is mass deworming? Paper has no relevant information"

Paper 3 title: "Effects of creatine supplementation on cognitive function of healthy individuals: A systematic review of randomized controlled trials"
Paper 3 abstract: "Background and aims: The aim of this systematic review is to investigate the effects of oral creatine administration on cognitive function in healthy individuals. Methods: A search of multiple electronic databases was performed for the identification of randomized clinical trials (RCTs) examining the cognitive effects of oral creatine supplementation in healthy individuals. Results: Six studies (281 individuals) met our inclusion criteria. Generally, there was evidence that short term memory and intelligence/reasoning may be improved by creatine administration. Performance on cognitive tasks stayed unchanged in young individuals. Conclusions: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear. It is imperative that creatine should be tested on patients with dementias or cognitive impairment."
Question 3 and answer 3: "What are the effects of creatine on cognition? Creatine supplementation may improve short-term memory and intelligence/reasoning"

Paper 4 title: "{title}"
Paper 4 abstract: "{abstract}"
Question 4 and answer 4: "{question}
""".strip()


claim_compress_prompt = """Concisely answer the question "How can I summarize long documents?" given the following context:

Context: "Inspired by how humans summarize long documents, we propose an accurate and fast summarization model that first selects salient sentences and then rewrites them abstractively (i.e., compresses and paraphrases) to generate a concise overall summary."

Question and answer: "How can I summarize long documents? Select salient sentences, then compress and paraphrase them"

====

Concisely answer the question "How can we make transformers work well with longer sequences?" given the following context:

Context: "Transformers do not scale very well to long sequence lengths largely because of quadratic self-attention complexity."

Question and answer: "How can we make transformers work well with longer sequences? Quadratic self-attention causes bad scaling"

====

Concisely answer the question "How does poverty affect the brain?" given the following context:

Context: "Building on a robust literature from animal models showing that environmental deprivation or enrichment shapes the brain, there has been increasing interest in understanding how the experience of poverty may shape the brain in humans."

Question and answer: "How does poverty affect the brain? In animal models environmental deprivation/enrichment shapes the brain"

====

Concisely answer the question "What are the effects of creatine on cognition?" given the following context:

Context: "The aim of this systematic review is to investigate the effects of oral creatine administration on cognitive function in healthy individuals."

Question and answer: "What are the effects of creatine on cognition? Reviews effects of oral creatine on cognition"

====

Concisely answer the question "Is longtermism correct?" given the following context:

Context: "In this paper, I will argue that the Stakes Principle is nonetheless false Furthermore, I will present a i a facie plausible view about the relationship between value and obligations in the context of the Non Identity Problem that suggests that axiological longtermism is true but deontic longtermism is false, even granting that the value at stake over the very long run is astronomical in comparison with the value at stake within the near term"

Question and answer: "Is longtermism correct? Axiological longtermism is true, deontic longtermism is false"

====

Concisely answer the question "{question}" given the following context:

Context: "{claim_text}"

Question and answer: "{question}
""".strip()

fast_claim_compress_prompt = """Question: {question}
Context: {claim_text}
Answer:"""


probabilistic_qa_prompt = """{abstract_lines}

Title:
{title}

Question:
{question}

Abstract answers question (yes/no/not sure):"""

from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class Paper:
    title: str
    abstract: str

neuro = [
    Paper(title="Single Cortical Neurons as Deep Artificial Neural Networks", abstract="""&lt;p&gt;We propose a novel approach based on modern deep artificial neural networks (DNNs) for understanding how the morpho-electrical complexity of neurons shapes their input/output (I/O) properties at the millisecond resolution in response to massive synaptic input. The I/O of integrate and fire point neuron is accurately captured by a DNN with a single unit and one hidden layer. A fully connected DNN with one hidden layer faithfully replicated the I/O relationship of a detailed model of Layer 5 cortical pyramidal cell (L5PC) receiving AMPA and GABAA synapses. However, when adding voltage-gated NMDA-conductances, a temporally-convolutional DNN with seven layers was required. Analysis of the DNN filters provides new insights into dendritic processing shaping the I/O properties of neurons. This work proposes a systematic approach for characterizing the functional &quot;depth&quot; of a biological neurons, suggesting that cortical pyramidal neurons and the networks they form are computationally much more powerful than previously assumed.&lt;/p&gt;""")
]
    
machine_learning = [ 
    Paper(title="Learning in High Dimension Always Amounts to Extrapolation", abstract="""
The notion of interpolation and extrapolation is fundamental in various fields from deep learning to function approximation. Interpolation occurs for a sample $x$ whenever this sample falls inside or on the boundary of the given dataset's convex hull. Extrapolation occurs when $x$ falls outside of that convex hull. One fundamental (mis)conception is that state-of-the-art algorithms work so well because of their ability to correctly interpolate training data. A second (mis)conception is that interpolation happens throughout tasks and datasets, in fact, many intuitions and theories rely on that assumption. We empirically and theoretically argue against those two points and demonstrate that on any high-dimensional ($>$100) dataset, interpolation almost surely never happens. Those results challenge the validity of our current interpolation/extrapolation definition as an indicator of generalization performances.
    """.strip()),
    Paper(title="ADOP: Approximate Differentiable One-Pixel Point Rendering", abstract="""
We present a novel point-based, differentiable neural rendering pipeline for scene refinement and novel view synthesis. The input are an initial estimate of the point cloud and the camera parameters. The output are synthesized images from arbitrary camera poses. The point cloud rendering is performed by a differentiable renderer using multi-resolution one-pixel point rasterization. Spatial gradients of the discrete rasterization are approximated by the novel concept of ghost geometry. After rendering, the neural image pyramid is passed through a deep neural network for shading calculations and hole-filling. A differentiable, physically-based tonemapper then converts the intermediate output to the target image. Since all stages of the pipeline are differentiable, we optimize all of the scene's parameters i.e. camera model, camera pose, point position, point color, environment map, rendering network weights, vignetting, camera response function, per image exposure, and per image white balance. We show that our system is able to synthesize sharper and more consistent novel views than existing approaches because the initial reconstruction is refined during training. The efficient one-pixel point rasterization allows us to use arbitrary camera models and display scenes with well over 100M points in real time.
    """.strip()),
    Paper(title="ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers", abstract="""
We describe a Question Answering (QA) dataset that contains complex questions with conditional answers, i.e. the answers are only applicable when certain conditions apply. We call this dataset ConditionalQA. In addition to conditional answers, the dataset also features: (1) long context documents with information that is related in logically complex ways; (2) multi-hop questions that require compositional logical reasoning; (3) a combination of extractive questions, yes/no questions, questions with multiple answers, and not-answerable questions; (4) questions asked without knowing the answers. We show that ConditionalQA is challenging for many of the existing QA models, especially in selecting answer conditions. We believe that this dataset will motivate further research in answering complex questions over long documents. Data and leaderboard are publicly available at url{https://github.com/haitian-sun/ConditionalQA}.
    """.strip()),
    Paper(title="QA Dataset Explosion: A Taxonomy of NLP Resources for Question Answering and Reading Comprehension", abstract="""
Alongside huge volumes of research on deep learning models in NLP in the recent years, there has been also much work on benchmark datasets needed to track modeling progress. Question answering and reading comprehension have been particularly prolific in this regard, with over 80 new datasets appearing in the past two years. This study is the largest survey of the field to date. We provide an overview of the various formats and domains of the current resources, highlighting the current lacunae for future work. We further discuss the current classifications of ``reasoning types" in question answering and propose a new taxonomy. We also discuss the implications of over-focusing on English, and survey the current monolingual resources for other languages and multilingual resources. The study is aimed at both practitioners looking for pointers to the wealth of existing data, and at researchers working on new resources.
    """.strip()),
    Paper(title="TLDR: Extreme Summarization of Scientific Documents", abstract="""
We introduce TLDR generation, a new form of extreme summarization, for scientific papers. TLDR generation involves high source compression and requires expert background knowledge and understanding of complex domain-specific language. To facilitate study on this task, we introduce SciTLDR, a new multi-target dataset of 5.4K TLDRs over 3.2K papers. SciTLDR contains both author-written and expert-derived TLDRs, where the latter are collected using a novel annotation protocol that produces high-quality summaries while minimizing annotation burden. We propose CATTS, a simple yet effective learning strategy for generating TLDRs that exploits titles as an auxiliary training signal. CATTS improves upon strong baselines under both automated metrics and human evaluations. Data and code are publicly available at https://github.com/allenai/scitldr.
    """.strip()),
          Paper(title="Want To Reduce Labeling Cost? GPT-3 Can Help", abstract="""Data annotation is a time-consuming and labor-intensive process for many NLP tasks. Although there exist various methods to produce pseudo data labels, they are often task-specific and require a decent amount of labeled data to start with. Recently, the immense language model GPT-3 with 175 billion parameters has achieved tremendous improvement across many few-shot learning tasks. In this paper, we explore ways to leverage GPT-3 as a low-cost data labeler to train other models. We find that, to make the downstream model achieve the same performance on a variety of NLU and NLG tasks, it costs 50% to 96% less to use labels from GPT-3 than using labels from humans. Furthermore, we propose a novel framework of combining pseudo labels from GPT-3 with human labels, which leads to even better performance with limited labeling budget. These results present a cost-effective data labeling methodology that is generalizable to many practical applications.""")
]

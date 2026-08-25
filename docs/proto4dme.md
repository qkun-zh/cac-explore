# Proto4DME: Interpretable Cell Counting via Additive Prototype Density Decomposition and Optimal-Transport Coverage

## Abdurahman Ali Mohammed

Iowa State University, USA

Wallapak Tavanapong

Iowa State University, USA

## Abstract

Cell counting via density map estimation pre- dicts a per-pixel density. Summing the den- sity yields the final count, a common read- out in clinical diagnostics and disease monitor- ing. Yet these models are often hard to audit when errors occur. We present Proto4DME, an interpretable density map estimator with faithful explanations by construction. The predicted density (and thus the count) is an additive, non-negative combination of contri- butions from learned visual patterns (proto- types). Prior prototype-based counting uses signed aggregation, which permits cancella- tion. In contrast, Proto4DME provides non- canceling attributions, in which increasing a prototype’s activation can only increase the pre- dicted density. So prototype heatmaps corre- spond to positive contributions for the count. Proto4DME learns spatial prototype activa- tion maps from backbone features and selects a compact set of prototypes using sparsity- inducing Hard-Concrete gates. To encourage diverse foreground coverage and prevent pro- totype collapse, we introduce an entropically- regularized optimal-transport coverage objec- tive. It allocates ground-truth density mass across prototypes under capacity constraints and induces competition among prototypes. Across three microscopy benchmarks (MBM, ADI, and DCC), Proto4DME achieves compet- itive mean absolute error (MAE) while produc- ing compact, auditable explanations that sup- port error analysis and model debugging.

## Data and Code Availability

All datasets used in this study are publicly available: MBM Kainz et al. (2015), ADI Paul Cohen et al. (2017), and DCC Marsden et al. (2018). We use the official dataset releases and the splits described [URL 🔗](#page-0)

abdu@iastate.edu

tavanapo@iastate.edu

in Sec.

[3.1.](#page-0)

Our implementation of Proto4DME is

publicly available at https://github.com/NRT-D4/ Proto4DME to facilitate reproducibility. [URL 🔗](https://github.com/NRT-D4/Proto4DME)

Institutional Review Board (IRB) This re- search does not involve data from live humans. It does not require IRB approval.

## 1. Introduction

Cell counting is the task of estimating how many cells are present in a microscopy field of view Guo et al. (2022). These counts are a routine quantitative read- out in biomedical studies (e.g., measuring prolifera- tion, comparing treatment conditions, or quantifying infection and immune response), and they often feed directly into downstream statistical analyses where small systematic biases can change conclusions. [URL 🔗](#page-0)

Automating cell quantification is challenging due to the nature of microscopy images. Images can range from sparsely populated fields to highly crowded re- gions with substantial overlap, and cells can vary widely in morphology and intensity, and appear un-

der uneven illumination or staining

Kainz et al. (2015). In many datasets, supervision is limited to sparse point annotations of cell centers rather than full instance masks, making classical de- tection or segmentation pipelines brittle. In addi- tion, background texture and debris can mimic cellu- lar structure, and acquisition settings can shift across batches. [URL 🔗](#page-0)

Density map estimation (DME) is therefore a com- mon strategy for counting in crowded scenes Lem- pitsky and Zisserman (2010). Instead of predict- ing discrete cell instances, a model predicts a den- sity map whose sum approximates the object count. This formulation is well-suited to biomedical imag- ing because it can be trained from point annota- tions, it tolerates overlap when boundaries are am- [URL 🔗](#page-0)

Xie et al. (2018); [URL 🔗](#page-0)


biguous, and it produces a spatial output that can be inspected visually Lempitsky and Zisserman (2010); Zhang et al. (2016). Modern density estimators build on fully convolutional backbones and multi-scale con- text. Crowd-counting architectures such as CSR- Net Li et al. (2018) are widely adopted. Vision Trans- former (ViT) backbones have also been adapted to microscopy counting, but typically rely on fine-tuning large pretrained models and can be comparatively resource-intensive to train and deploy Mohammed et al. (2025a); Lin et al. (2022). [URL 🔗](#page-0)

Even when density map estimators are accurate on average, they can be difficult to trust and de- bug. When predicted counts deviate from expecta- tions, practitioners need to know whether the model is mistaking debris for cells, missing dim or small cells, or failing in particular tissue regions. This re- quirement is not only scientific but also operational. In automated image-analysis pipelines, practitioners often need actionable feedback Wang et al. (2024) about failure modes before a model can be deployed in practice Rudin (2019). [URL 🔗](#page-0)

Recently, CountXplain Mohammed et al. (2025b) introduced prototype-based explanations for mi- croscopy density map estimation. However, aspects of its formulation can hinder faithful attribution in counting. Prototype coefficients may be negative, so highlighted regions can represent negative contribu- tions and reduce the predicted density via cancella- tion. Moreover, its extreme-point supervision pro- vides limited coverage of diverse cell appearances, and the prototype set size must be specified in ad- vance rather than selected automatically. [URL 🔗](#page-0)

We address these challenges with Proto4DME, an interpretable density map estimator with faithful explanations by construction. Our main contribu- tions are:

- Optimal-transport prototype coverage: an entropically regularized optimal-transport objec- tive that allocates ground-truth density mass across prototypes under capacity constraints, en- couraging diverse foreground coverage and pre- venting collapse.

- Non-negative density head: a positivity con- strained 1 × 1 density head that maps proto- type similarity maps to a final density map. This yields an exact additive, non-cancelling de- composition of the predicted density (and total count).

- Automatic prototype selection for density estimation: an adaptation of Hard-Concrete gates Louizos et al. (2017) to density map es- timation, enabling the model to automatically select a compact, data-driven set of prototypes for each dataset. [URL 🔗](#page-0)

We evaluate Proto4DME on three microscopy bench- marks, MBM Kainz et al. (2015), DCC Marsden [URL 🔗](#page-0)

et al. (2018), [URL 🔗](#page-0)

and ADI

[Paul Cohen et al. (2017).](#page-0)

Proto4DME achieves competitive mean absolute er- ror (MAE) while producing compact, auditable ex- planations that support error analysis and model de- bugging.

In the remainder of the paper, we describe the Proto4DME architecture and training objec- tive, present quantitative results and ablations, and demonstrate how the learned prototypes support global summarization of counting concepts as well as localized, faithful explanations of individual predic-

tions.

## 2. Related Work

Prior OT-based counting methods improve accuracy by matching predicted and target spatial mass distri- butions (including unbalanced variants for mass mis- match) in crowd and cell counting Wang et al. (2020); Babu Sam et al. (2022); Ma et al. (2021); Ding et al. (2023). In our work, OT is only an auxiliary reg- ularizer; the core contribution is a prototype-based density estimator with gated prototype selection and per-prototype similarity maps that yield faithful local explanations and compact concept sets. [URL 🔗](#page-0)

Post hoc interpretability methods offer partial in- sight but have important limitations in this set- ting. Class Activation Mapping Zhou et al. (2016), Grad-CAM Selvaraju et al. (2017), and pixel-wise attribution methods Bach et al. (2015); Shrikumar et al. (2017); Lundberg and Lee (2017) can fail san- [URL 🔗](#page-0)

ity checks

heatmaps that neither decompose the final integral count nor map cleanly to reusable biological concepts (cell morphologies, imaging artifacts, or background patterns).

Prototype-based interpretability provides a natu- ral bridge. Self-explainable prototype models jus- tify predictions via similarity to learned prototypi- cal patterns, and prior work (e.g., ProtoPNet) shows that prototypes can ground human-understandable reasoning in representative parts Chen et al. (2020); [URL 🔗](#page-0)

Adebayo et al. (2018) [URL 🔗](#page-0)

and typically yield


*Figure 1: (A) Black-box vs (B) Interpretable Density*

*Map Estimation models*

Rymarczyk et al. (2021); Chen et al. (2019); Bar- nett et al. (2021); Mohammadjafari et al. (2021); Singh and Yow (2021a,b); Djoumessi et al. (2024). While regression variants exist Hesse and Namburete (2022); Hesse et al. (2024), density map estimation adds a spatial requirement. Explanations must spec- ify where attributions appear and how they aggregate into the final count, while remaining faithful by con- struction and compact so each prediction can be com- municated by a small number of dominant concepts Molnar (2022). [URL 🔗](#page-0)

CountXplain Mohammed et al. (2025b) is the first prototype-based approach for interpretable density map estimation in microscopy. It learns cell and back- ground prototypes and predicts density as a learned linear combination of prototype similarity maps. [URL 🔗](#page-0)

While CountXplain establishes an important di- rection, several design choices weaken the link be- tween prototype visualizations and faithful, action- able counting explanations. First, its density head is unconstrained and can assign negative coeffi- cients, which enables cancellation. A prototype may strongly activate in a region yet reduce the predicted density and the final count, so prototype heatmaps do not allow a simple additive interpretation of con- tribution.

Second, its prototype-to-feature alignment is su- pervised only at two extreme spatial locations per im- age, namely the max-density location for “cell” and the min-density location for “background”. In dense and heterogeneous scenes, this provides limited sig- nal and does not encourage prototypes to collectively cover the full foreground density mass.

Third, the number of prototypes is fixed a pri- ori with no mechanism to deactivate redundant pro-

*Table 1: Dataset statistics.*

| Dataset | Image Size | Ntrain/Ntotal | Cell Count |
| --- | --- | --- | --- |
| MBM Kainz et al. (2015) | 600 × 600 | 15/44 | 126 ± 33 |
| ADI Paul Cohen et al. (2017) | 150 × 150 | 50/200 | 165 ± 44 |
| DCC Marsden et al. (2018) | varied | 100/176 | 34 ± 22 |

totypes during training. Redundant concepts can therefore persist and dilute interpretability.

## 3. Methods

## 3.1. Datasets

We evaluate Proto4DME on three microscopy cell counting benchmarks: MBM, DCC, and ADI. Each dataset provides microscopy images paired with sparse cell annotations (point marks at cell centers or equivalent) from which ground-truth density maps can be constructed. Together, these benchmarks span heterogeneous imaging conditions, diverse cell morphologies, and a wide range of cell densities.

MBM. We use the Modified Bone Marrow (MBM) dataset introduced by Cohen et al. Paul Cohen et al. (2017), derived from the BM dataset of Kainz et al. Kainz et al. (2015). The original BM data consist of real bone marrow images from healthy individu- als, with standard staining that depicts nuclei in blue while other constituents appear in shades of pink/red. [URL 🔗](#page-0)

ADI. The human subcutaneous adipose tissue (ADI) dataset was constructed from the Genotype- Tissue Expression (GTEx) Consortium Lonsdale et al. (2013) and contains densely packed adipocyte cells. Regions of interest (ROIs) are sampled from high-resolution histology slides using a sliding win- dow and then downsampled to a suitable scale by Paul Cohen et al. (2017). Adipocytes vary substan- tially in size (approximately 20–200 µm) and are often tightly packed with few gaps, making ADI a challeng- ing test case for automated cell counting. [URL 🔗](#page-0)

DCC. Marsden et al. Marsden et al. (2018) built the Dublin Cell Counting (DCC) dataset to represent a wide range of cells, including embryonic mouse stem cells, human lung adenocarcinoma, and human mono- cytes. Image sizes range from 306×322 to 798×788, increasing dataset variation. [URL 🔗](#page-0)

## 3.2. Model Architecture

Proto4DME is a prototype-based density map esti- mator built on top of a fully convolutional counting


backbone. This choice provides spatial inductive bi- ases and computational efficiency, promoting robust generalization in the data-scarce regime common in cell counting. Given an input image I ∈ RH×W×Cin , where Cin, W, and H are the channel, width, and height, respectively, the model produces (i) a pre- dicted density map ˆD and (ii) a set of per-prototype activation maps that provide a faithful, spatially lo- calized explanation of the prediction.

Backbone feature extractor. We adopt a CSRNet-style architecture with a VGG Simonyan and Zisserman (2015) like front-end and a dilated convolution back-end to preserve spatial resolution while providing a large effective receptive field. The backbone maps the image to a mid-level feature ten- sor [URL 🔗](#page-0)

Here, C is the number of backbone feature channels and (H′,W′) are the backbone output spatial dimen- sions. We use the feature channels produced by the CSRNet back-end and apply a lightweight 1×1 “add- on” transformation (feature enhancer) to obtain a bounded feature representation. In the add-on, σ(·) denotes the sigmoid nonlinearity (so F˜ is bounded el- ementwise in [0, 1]), ∗ is convolution, and Wadd and badd are the 1×1 convolution weights and bias. This add-on stack stabilizes prototype matching by con- straining feature magnitudes.

Prototype bank and distance maps. Proto4DME maintains a bank of K learned prototypes {pk}K k=1, where each pk ∈ RC represents a recurrent visual pattern in the backbone feature space. Because background prototypes are typically not used by the counting head and are redundant under our non-negative additive formulation, we use only cell prototypes. Background is then implicitly defined as regions with uniformly low cell-prototype evidence (i.e., low activation across all prototypes). For every spatial location (i, j), we compute the squared Euclidean distance between the local feature vector and each prototype:

where gk ∈ [0, 1] is the gate value defined below yielding a distance tensor Φ ∈ RK×H′×W′ . In prac- tice, distances are computed efficiently using an L2- convolution following Chen et al. (2019). [URL 🔗](#page-0)

Similarity mapping. Following prior work Chen et al. (2019), we convert distances to prototype simi- larity maps Sk(i, j) using a monotone log transform. The backbone feature extractor, squared-ℓ2 pro- totype matching, and distance-to-similarity mapping follow standard prototype-learning practice. We next [URL 🔗](#page-0)

introduce components specific to Proto4DME.

Hard-Concrete gates for prototype selection. To avoid committing to a fixed number of proto- types, we associate each prototype with a gate. Let gk ∈ [0, 1] denote the open probability of prototype k under a Hard-Concrete relaxation Louizos et al. (2017). During inference, we use the deterministic expectation gk. As used in the distance computa- tion, we gate prototype vectors via ˜pk = gk pk. We additionally gate each similarity channel so that pro- totypes with closed gates are silent: [URL 🔗](#page-0)

Above-baseline evidence (excess activation). Raw similarity maps can contain a non-zero baseline due to feature normalization and the log transform. To make activations reflect above-baseline evidence, we center each prototype similarity map by its spatial mean and retain only positive deviations:

The resulting maps Ek can be interpreted as prototype-specific heatmaps highlighting locations where prototype k matches more strongly than its image-level baseline.

Non-negative 1 × 1 density head. Finally, the predicted density map at the backbone resolution is obtained as a non-negative linear combination of per- prototype activation maps:

We enforce wk ≥ 0 by parameterizing the 1×1 convo- lution weights with a positivity constraint (e.g., soft- plus).

Faithful additive decomposition. The predicted count is obtained by summing the density map over space:


*Figure 2: Overview of Proto4DME and pruning. The input image is encoded by a fully convolutional*

*backbone and feature adapter. Prototypes are matched to spatial features via squared L2 distance to form distance maps that are converted to similarity maps. Hard-Concrete gates modulate prototype channels during training. During pruning, prototypes with gates below an automatically determined threshold are removed. A non-negative density head maps prototype similarities to the predicted density map.*

Substituting the density-head expression yields an ex- act additive decomposition into per-prototype contri- butions:

Because wk ≥ 0 and Ek(i, j) ≥ 0, contributions are non-negative and cannot cancel. This makes expla- nations directly comparable across images.

## 3.3. Training Objective

Proto4DME is trained end-to-end with (i) a density regression loss, (ii) an image-level counting loss on global count, (iii) a prototype coverage and com- petition term via entropic optimal transport (OT), and (iv) an ℓ0-style sparsity regularizer on prototype gates.

Density map regression (Huber) and image- level counting Given a ground-truth density map D, the model predicts a density Dˆ ∈ RH′×W′ . We

supervise the density map with a Huber (Smooth-ℓ1)

loss,

where β is the Huber transition point. In addition, we explicitly constrain the total predicted count by penalizing the absolute error between summed densi- ties:

This term directly optimizes image-level counting ac- curacy while Ldens preserves spatial supervision.

Prototype coverage and competition via bal- anced assignment Unlike CountXplain, which aligns prototypes using extreme points per image (maximum density for cell and minimum density for background), we use entropically regularized optimal transport for balanced assignment. For each image, we treat the ground truth density map D(i, j) as a distribution of foreground mass over spatial locations by normalizing it to sum to one,


where δ > 0 is a small constant for numerical stabil- ity. We then flatten tij into a vector t ∈ RH′W′ using a single spatial index p ∈ {1, . . . ,H′W′} correspond- ing to location (i, j), and denote the resulting entries by tp. Images with P u,v D(u, v) = 0 provide no valid unit-mass target distribution, so we skip them (i.e., no OT loss is computed for that image).

Given prototype distance maps Φk(i, j) computed from the backbone features, we form the transport cost as the raw distances, flattened consistently as

where p is the flattened index of (i, j). Thus Q ∈ RK×H′W′ and t ∈ RH′W′ share the same spatial in- dexing.

Prototype capacities are determined by the Hard Concrete gates. Recall gk ∈ [0, 1] is the expected open probability of prototype k. We convert these gate values into a probability distribution over prototypes using a temperature-controlled log softmax Hinton et al. (2015) [URL 🔗](#page-0)

where τ is the gate temperature, ϵgate prevents log(0), and ϵ0 is a small threshold used to trig- ger a uniform fallback when all gates are effectively closed. All three are fixed positive constants. We set ϵgate = 10−8 purely for numerical stability and

Puse k gk ϵ0

= 10−8 to detect the degenerate case where

is effectively zero, in which case we fall back

to a uniform capacity allocation. The temperature τ > 0 controls the softness of the capacity distri- bution; we use a moderate value (τ = 0.65). This construction yields simplex marginals required by op- timal transport (nonnegative and summing to one), allocates more capacity to prototypes that are more likely to be active, and still enforces competition be- cause increasing mass for one prototype necessarily reduces it for others.

We then solve for a transport plan Π ∈ RK×H′W′ via Sinkhorn iterations Cuturi (2013) [URL 🔗](#page-0)

Π⋆ = arg min ⟨Π, Q⟩ + εot X Πk,p [URL 🔗](#page-0)

where p indexes spatial locations, εot is the entropic regularization coefficient, and the constraints ensure that all foreground mass t is assigned while proto- types compete under the capacity distribution a. The

OT loss is defined as the expected transport cost un- der the optimal plan,

Because t

is derived from the ground truth density,

this term forces coverage of foreground regions, and because a is gate-controlled, it discourages multiple prototypes from explaining the same locations and promotes specialization.

L0 sparsity regularization We regularize the ex- pected number of active prototypes using the analytic expected L0 penalty induced by Hard Concrete gates,

Total loss Our training objective combines stan- dard density/count supervision with two new regu- larizers, an OT-style coverage/competition loss and an L0 gate sparsity loss. We minimize the weighted sum:

All coefficients in (18) are treated as hyperparameters [URL 🔗](#page-0)

and selected on a validation set.

Optional post-training pruning and head refit- ting The Hard-Concrete regularizer encourages the model to rely on a small subset of prototypes, but in practice, a few prototypes may retain small gate probabilities and contribute negligibly to the pre- dicted density. For deployment and improved inter- pretability, we optionally apply a post-training prun- ing step that removes such weak prototypes and pro- duces a compact model with fewer heatmaps to in- spect. We rank prototypes by an effective contribu- tion score rk that combines the non-negative density- head weight and the expected gate openness. Re- call that wk ≥ 0 is the density-head coefficient and gk ∈ [0, 1] denotes the expected gate openness for prototype k. We define

We select the keep set using Algorithm 1, steps 1–5 (score computation, knee thresholding Satopaa et al. (2011), and safeguards for minimum retention). [URL 🔗](#page-0)

We then construct the pruned model using Algo- rithm 1, steps 6–8 (dropping prototypes, rewiring the head, and transferring parameters so that retained prototypes preserve their learned influence). [URL 🔗](#page-0)


Finally, because pruning can still introduce small deviations due to numerical effects and the removal of stochastic gating during training, we perform a short refitting stage as part of pruning in which the pruned model is distilled to match the unpruned model’s den- sity predictions on a calibration set. In this stage, we keep the backbone and prototypes fixed and optimize only the density head. This lightweight procedure typically preserves counting accuracy while improv- ing interpretability and reducing inference cost.

To summarize, Proto4DME couples prototype ex- planations with density estimation in a single model. Prior work uses gating mainly for sparsity and model compression, such as pruning weights or channels Louizos et al. (2017). We instead apply Hard- Concrete gates to the prototypes, enabling the model to learn a compact, data-driven set of visual concepts. We also change how prototypes are trained. While optimal transport is often used for matching or dis- tribution alignment, we use balanced OT assignment to the ground-truth density mass. This enforces cov- erage of density and competition between prototypes, reducing redundancy and collapse. Combined with a non-negative density head, Proto4DME provides faithful, non-canceling additive explanations of both the density map and the total count. [URL 🔗](#page-0)

## 4. Results

## 4.1. Experimental setup and evaluation metrics

We evaluate Proto4DME on three microscopy cell counting benchmarks: MBM, ADI, and DCC (Sec. 3.1). For each dataset, we report image-level counting error using mean absolute error (MAE) be- tween the predicted count ˆn and the ground-truth count n. Counts are obtained by summing the pixels of the predicted density map. [URL 🔗](#page-0)

We compare Proto4DME to (i) a strong non- interpretable density regression baseline using the same counting backbone (CSRNet), and (ii) the closest prototype-based density estimator baseline (CountXplain). For each trial, we randomly sam- ple train/test splits for each dataset and evaluate all methods on identical splits; we repeat this procedure 5 times and report the mean and standard deviation of MAE.

We use K = 10 cell prototypes and Hard-Concrete gating initialized with p0 = 0.8 and temperature τ = 0.65. We optimize with Adam (lr = 5 × 10−3,

## Algorithm 1: Optional post-training prototype pruning and head refitting

Input: Trained prototype model M with prototypes {p}K k=1, non-negative head weights {wk}K k=1, gate expecta- tions {gk}K k=1, minimum keep Kmin ≥ 1

Output: Pruned model M′ with fewer prototypes and com- parable predictions

- 1. Compute prototype scores rk ← wkgk for k = 1, . . . ,K

- 2. Normalize scores to [0, 1], sort them in ascending order, and select a threshold γ using Kneedle algorithm Satopaa et al. (2011) [URL 🔗](#page-0)

- 3. Define keep indices K ← {k : rk ≥ γ}

- 4. If |K| = K, set K to the indices of the top max(Kmin, 1) scores

- 5. If |K| < Kmin, expand K to include the top Kmin indices

- 6. Construct the pruned model M′ by keeping only proto- types {pk}k∈K and reinitializing the head to have |K| input channels, then transfer the corresponding head weights for k ∈ K.

- 7. Set the gate parameters of M′ so that its gate expecta- tions match {gk}k∈K

- 8. Freeze backbone and prototype parameters of M′

- 9. For Z refit steps:

- (a) Sample a minibatch I

- (b) Compute teacher density DˆT ← M(I) and global count ˆnT ← P ˆDT

- (c) Compute student density DˆS ←M′(I) and global count ˆnS ← P ˆDS

- (d) Update only the head ofM′ to minimize (ˆnS−ˆnT)2

10. Return M′

batch size 16) for up to 1000 epochs with early stop- ping (patience 150). The total loss uses λcnt = 1, λdens = 10, λot = 5, and λ0 = 5×10−4. OT is solved with Sinkhorn iterations (40) and entropic regular- ization εot = 0.08. We set β = 0.5, δ = 10−12, and Kmin = 2, and we use H′ = 8H 1 and W′ = 1 8W. Loss weights and OT parameters were selected us- ing a small validation-based search. The remaining training settings follow standard practice and were kept fixed.

## 4.2. Counting results

Table 2 reports mean MAE on MBM, ADI, and DCC. Proto4DME achieves the best mean MAE on MBM and DCC, improving over CSRNet by 3.7% and 21.8%, respectively, and over CountXplain by 14.1% and 21.2%. On ADI, Proto4DME improves over CSRNet (10.8 ± 2.77 vs. 12.37 ± 0.64) but remains behind CountXplain (7.71 ± 2.34). [URL 🔗](#page-0)


*Table 2: Counting results (MAE↓) (mean ± std.) on*

*MBM, ADI, and DCC.*

| Method | MBM ADI | DCC |
| --- | --- | --- |
| CSRNet | 6.19 ± 0.53 12.37 ± 0.64 2.61 ± 0.31 |   |
| CountXplain 6.94 ± 1.74 | 7.71 ± 2.34 | 2.59 ± 0.23 |
| Proto4DME 5.96 ± 0.68 | 10.8 ± 2.77 | 2.04 ± 0.26 |

Note: Proto4DME results are reported for the pruned model (Alg. 1). [URL 🔗](#page-0)

While CountXplain attains a lower MAE on ADI, this MAE gap (10.80 vs. 7.71) must be weighed against the cost of explanation faithfulness. The per- formance gap likely reflects the dataset’s inherent difficulty rather than a flaw in Proto4DME’s non- negative head. Specifically, adipocytes in the ADI dataset are highly heterogeneous and tightly packed, ranging from 20 to 200 µm in size. They feature pale, near-empty interiors, and only their thin outer mem- branes are stained. This distinct morphology pro- duces extremely subtle boundaries in dense scenes, making it challenging for spatial models to isolate in- dividual cell instances without broader context. Con- sequently, overlapping local evidence creates signifi- cant ambiguity during density map estimation.

To handle this ambiguity, CountXplain relies heav- ily on a cancellation mechanism at the cost of faith- fulness. As detailed in Table 3, CountXplain as- signs negative coefficients to 60% of its prototypes on ADI, resulting in a negative-mass ratio of 0.433. This means that strong activation of a prototype can actually reduce the predicted count. As visi- ble in Figure 3 (ADI row), CountXplain’s prototype maps (e.g., Proto 1–Proto 3) tend to activate diffusely over broad non-cell tissue regions—areas where the ground-truth dot annotation marks only the sparse cell centers. The signed density head then compen- sates for these massive spurious responses by assign- ing negative coefficients to subtract the excess den- sity. While this unconstrained regression approach recovers overall counting accuracy, it fundamentally breaks additive interpretability. For a practitioner auditing the model, a brightly highlighted heatmap region does not reliably correspond to a counted cell; instead, it may represent a background region where the model is aggressively suppressing its own overpre- dictions. Thus, the observed performance gap on ADI represents a faithfulness-accuracy tradeoff by design. [URL 🔗](#page-0)

In contrast, Proto4DME maintains strict additive interpretability. The guarantee that wk ≥ 0 ∀k en-

sures that every prototype heatmap is a valid positive attribution. This is a crucial structural requirement for auditable counting in biomedical applications, not merely an aesthetic choice. Forced to construct the density map through pure addition, Proto4DME pro- duces more consistently cell-aligned activation pat- terns across its top prototypes. Because contribu- tions are strictly non-negative, practitioners can read these maps directly as additive spatial components that build up the predicted count (Figure 3). Fur- thermore, regarding the observed instability on ADI, the dataset’s densely packed cells and subtle bound- aries create highly overlapping local evidence. This overlap makes learning a strictly non-negative spatial decomposition inherently more sensitive to different training runs. Notably, CountXplain also exhibits substantial variance on ADI (±2.34) despite having the freedom to use negative cancellation. This shared instability suggests that the variance primarily re- flects the intrinsic difficulty of the ADI dataset rather than a modeling deficiency. [URL 🔗](#page-0)

Overall, these results indicate that Proto4DME can match or improve upon a strong density-regression baseline, and it remains competitive with the closest prototype-based baseline. In the following sections, we analyze Proto4DME components and explanation quality.

## 4.3. Local explanations: per-image prototype contribution breakdown

Figure 3 illustrates representative local explana- tions from Proto4DME by visualizing the input im- age alongside the top-K prototype similarity maps, which highlight regions most similar to each selected prototype. [URL 🔗](#page-0)

Qualitatively, this decomposition enables targeted auditing: overcounting cases often coincide with strong activation from prototypes that respond to debris-like texture, while undercounting cases show missing activation on dim or small cells. Because con- tributions are non-negative, prototype heatmaps can be interpreted directly as attribution for density, and the stacked contribution totals correspond exactly to the model’s predicted count.

## 4.4. Global interpretability: learned prototypes

We visualize learned prototypes to evaluate whether Proto4DME discovers recurring, human- interpretable patterns that align with microscopy


## Proto4DME

*Figure 3: Local prototype explanations on three test images. Left: input and ground-truth density (with*

*total count). Right: prototype similarity maps for CountXplain (top) and Proto4DME (bottom); brighter regions indicate higher similarity/activation (computed from distances in Eq. (3)). [URL 🔗](#page-0)*

*Figure 4: Model’s global knowledge captured by pro-*

*totypes. For each prototype, we display the three most representative image re- gions from the datasets (ranked by proto- type similarity).*

structure. For each prototype, we retrieve its top-3 global exemplars by scanning the training set and selecting the spatial locations that minimize the prototype-feature distance (equivalently, maximize similarity). We then extract an adaptive patch around the activated region by thresholding the upsampled prototype activation map and using connected components to obtain a tight bounding box. Figure 4 presents the resulting exemplar gallery (three patches per prototype), enabling direct inspection of the visual patterns each concept captures. [URL 🔗](#page-0)

Across datasets, the learned prototypes correspond to distinct cell appearances and local contexts (e.g., compact bright nuclei, dim smaller cells, and clus- tered structures), while also capturing recurrent con- founders such as textured background and debris-like


artifacts. Finally, the sparsity-inducing gate mecha- nism suppresses redundant prototype channels, yield- ing a compact set of concepts and a clearer global prototype gallery for practitioner inspection.

## 4.5. Ablations and analysis of Proto4DME components

Rather than treating ablations as purely accuracy- oriented variants, we evaluate each Proto4DME com- ponent against the specific principle it is designed to enforce. These principles include non-negative, cancellation-free attribution, prototype coverage and specialization through OT, and automatic concept set selection through sparsity-inducing gates. We there- fore report targeted diagnostics alongside counting errors.

## 4.5.1. Signed aggregation breaks additive prototype contributions

Proto4DME constrains the density head to be non- negative, so the predicted density decomposes addi- tively into per-prototype contributions. In contrast, CountXplain aggregates prototype similarity maps using an unconstrained signed head. Inspecting the learned coefficients reveals that CountXplain assigns negative weights on all three datasets (Table 3), im- plying that stronger activation of some prototypes can reduce the predicted density. This enables can- cellation and complicates the interpretation of pro- totype heatmaps, since high activation does not nec- essarily correspond to a positive contribution to the final count. Proto4DME avoids this behavior by con- struction via a non-negative density head. [URL 🔗](#page-0)

*Table 3: Signed vs. non-negative density-head co-*

*efficients reported as a percentage of neg- Pative weights and the negative-mass ratio wi<0 |wi|/ P i |wi| reported.*

|   | ADI DCC MBM |
| --- | --- |
| CountXplain: % wi < 0 ↓ | 60.0 40.0 60.0 |
| CountXplain: neg-mass ↓ | 0.433 0.315 0.569 |
| Proto4DME: % wi < 0 ↓ | 0.0 0.0 0.0 |
| Proto4DME: neg-mass ↓ | 0.000 0.000 0.000 |

*Table 4: Pruning compactness across datasets (5*

*runs). Kkept is after Alg. 1; MAE is re- ported pre→post pruning+refit. Numbers in parentheses indicate the min–max range of Kkept over runs. [URL 🔗](#page-0)*

| Dataset | Kkept/Ktotal | MAE (pre→post) |
| --- | --- | --- |
| DCC 5.0 ± 0.7/10 (4–6) 1.93 ± 0.30 → 2.04 ± 0.26 |   |   |
| ADI 7.6 ± 1.9/10 (5–9) 10.01 ± 1.63 → 10.80 ± 2.77 |   |   |
| MBM 3.4 ± 0.5/10 (3–4) 5.83 ± 0.72 → 5.96 ± 0.68 |   |   |

## 4.5.2. Balanced assignment (OT) improves coverage and prototype diversity

We next study the effect of the balanced assignment objective. Removing OT does not change the ex- planation mechanism (prototypes and non-negative aggregation remain), but it weakens the pressure for prototypes to collectively cover the foreground den- sity mass. To quantify this effect, we measure the mean minimum prototype distance on foreground pixels (dfg min) and background pixels (dbg min) across the validation set, where foreground is defined by nonzero ground-truth density. We summarize separation us-

ing ∆dmin = dbg min − dfg min, and additionally report a

min−dfg

scale-invariant relative gap ∆rel = dbg

min to ac-

min+dfg

dbg

min

count for potential global rescaling across ablations. With OT enabled, prototypes are closer to fore- ground than background (dfg min = 3.17, dbg min = 3.55, ∆dmin = +0.38, ∆rel = +0.056). In contrast, re- moving OT reverses this relationship (dfg min = 20.18, dbg min = 15.40, ∆dmin = −4.78, ∆rel = −0.134), indi- cating that prototypes drift toward background pat- terns without an explicit coverage and competition constraint. See Appendix A for qualitative results. [URL 🔗](#page-0)

## 4.5.3. Gating produces compact concept sets.

We quantify how gating translates into a discrete, compact prototype set using the optional post- training pruning procedure (Alg. 1), which selects prototypes via the combined score rk = wkgk and re- fits only the counting head. Table 4 shows that prun- ing yields consistent compression across datasets, but with dataset-dependent retained set sizes. On DCC, half of the prototypes are kept on average out of five runs (5.0 ± 0.7 out of 10). ADI retains a larger subset (7.6 ± 1.9), suggesting higher concept diversity/coverage requirements. Conversely, MBM [URL 🔗](#page-0)


prunes more aggressively (3.4 ± 0.5). Despite these reductions, MAE is generally preserved after refitting (small average change on DCC/MBM), while ADI ex- hibits higher variance with occasional larger degra- dation when fewer prototypes are retained. Because pruning uses rk = wkgk, this mainly measures which prototypes have both higher gate probability (high gk) and larger head weights (high wk), rather than the gates alone.

## 5. Conclusion

We presented Proto4DME, an interpretable den- sity map estimator for cell counting with faithful explanations by construction. A non-negative den- sity head yields an exact, additive decomposition of both the predicted density and the final count into per-prototype contributions, while an entropi- cally regularized optimal-transport coverage objec- tive and Hard-Concrete gating promote diverse fore- ground coverage and compact concept sets.

Because prototypes are learned without concept la- bels, they may not align with biologically meaningful categories. So concept validation/curation remains future work. Moreover, Proto4DME inherits biases from density supervision (e.g., kernel choice and an- notation noise), which can distort learned prototypes and explanations.

## Acknowledgments

This work is partially supported by the National Sci- ence Foundation under Grant No. 2152117. Any opinions, findings, and conclusions or recommenda- tions expressed in this material are those of the au- thor(s) and do not necessarily reflect the views of the National Science Foundation.

## References

Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. San- ity checks for saliency maps. Advances in neural information processing systems, 31, 2018.

Deepak Babu Sam, Abhinav Agarwalla, Jimmy Joseph, Vishwanath A Sindagi, R Venkatesh Babu, and Vishal M Patel. Completely self-supervised crowd counting via distribution matching. In Euro- pean Conference on Computer Vision, pages 186– 204. Springer, 2022.

Sebastian Bach, Alexander Binder, Gr´egoire Mon- tavon, Frederick Klauschen, Klaus-Robert M¨uller, and Wojciech Samek. On Pixel-Wise Explana- tions for Non-Linear Classifier Decisions by Layer- Wise Relevance Propagation. PLOS ONE, 10(7): e0130140, July 2015. ISSN 1932-6203.

Alina Jade Barnett, Fides Regina Schwartz, Chaofan Tao, Chaofan Chen, Yinhao Ren, Joseph Y Lo, and Cynthia Rudin. A case-based interpretable deep learning model for classification of mass le- sions in digital mammography. Nature Machine Intelligence, 3(12):1061–1070, 2021.

Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This Looks Like That: Deep Learning for Interpretable Image Recognition. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Zhi Chen, Yijie Bei, and Cynthia Rudin. Concept whitening for interpretable image recognition. Na- ture Machine Intelligence, 2(12):772–782, Decem- ber 2020. ISSN 2522-5839.

Marco Cuturi. Sinkhorn distances: Lightspeed com- putation of optimal transport. Advances in neural information processing systems, 26, 2013.

Yuanyuan Ding, Yuanjie Zheng, Zeyu Han, and Xinbo Yang. Using optimal transport theory to optimize a deep convolutional neural network mi- croscopic cell counting method. Medical & Biolog- ical Engineering & Computing, 61(11):2939–2950, 2023.

Kerol Djoumessi, Bubacarr Bah, Laura K¨uhlewein, Philipp Berens, and Lisa Koch. This actu- ally looks like that: Proto-bagnets for local and global interpretability-by-design. In Interna- tional Conference on Medical Image Computing and Computer-Assisted Intervention, pages 718– 728. Springer, 2024.

Yinong Guo, Chen Wu, Bo Du, and Liangpei Zhang. Density Map-based vehicle counting in remote sensing images with limited resolution. ISPRS Journal of Photogrammetry and Remote Sensing, 189:201–217, July 2022. ISSN 0924-2716.

Linde S. Hesse and Ana I. L. Namburete. INSightR- Net: Interpretable Neural Network for Regres- sion Using Similarity-Based Comparisons to Pro- totypical Examples. In Linwei Wang, Qi Dou,


## Proto4DME

P. Thomas Fletcher, Stefanie Speidel, and Shuo Li, editors, Medical Image Computing and Com- puter Assisted Intervention – MICCAI 2022, Lec- ture Notes in Computer Science, pages 502–511, Cham, 2022. Springer Nature Switzerland. ISBN 9783031164378.

Linde S. Hesse, Nicola K. Dinsdale, and Ana I. L. Namburete. Prototype learning for explainable brain age prediction. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 7903–7913, Jan- uary 2024.

Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In NIPS Deep Learning and Representation Learning Workshop, 2015. URL http://arxiv.org/abs/ 1503.02531. [URL 🔗](http://arxiv.org/abs/1503.02531)

Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, and Vincent Lepetit. You Should Use Regression to Detect Cells. In Nassir Navab, Joachim Hornegger, William M. Wells, and Ale- jandro F. Frangi, editors, Medical Image Comput- ing and Computer-Assisted Intervention – MIC- CAI 2015, Lecture Notes in Computer Science, pages 276–283, Cham, 2015. Springer International Publishing. ISBN 978-3-319-24574-4.

Victor Lempitsky and Andrew Zisserman. Learning To Count Objects in Images. In Advances in Neural Information Processing Systems, volume 23. Cur- ran Associates, Inc., 2010.

Yuhong Li, Xiaofan Zhang, and Deming Chen. CSR- Net: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1091–1100, June 2018. ISSN: 2575-7075.

Hui Lin, Zhiheng Ma, Rongrong Ji, Yaowei Wang, and Xiaopeng Hong. Boosting crowd counting via multifaceted attention. In Proceedings of the IEEE/CVF conference on computer vision and pat- tern recognition, pages 19628–19637, 2022.

John Lonsdale, Jeffrey Thomas, Mike Salvatore, Re- becca Phillips, Edmund Lo, Saboor Shad, Richard Hasz, Gary Walters, Fernando Garcia, Nancy Young, et al. The genotype-tissue expression (gtex) project. Nature genetics, 45(6):580–585, 2013.

Joseph Paul Cohen, Genevieve Boucher, Craig A. Glastonbury, Henry Z. Lo, and Yoshua Ben- gio. Count-ception: Counting by Fully Convolu- tional Redundant Counting. In Proceedings of the

Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l 0 regularization. arXiv preprint arXiv:1712.01312, 2017.

Scott M. Lundberg and Su-In Lee. A unified ap-

proach to interpreting model predictions. In

Pro-

ceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 4768–4777, Red Hook, NY, USA, 2017. Cur- ran Associates Inc. ISBN 9781510860964.

Zhiheng Ma, Xing Wei, Xiaopeng Hong, Hui Lin, Yunfeng Qiu, and Yihong Gong. Learning to count via unbalanced optimal transport. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 2319–2327, 2021.

Mark Marsden, Kevin McGuinness, Suzanne Little, Ciara E. Keogh, and Noel E. O’Connor. Peo- ple, Penguins and Petri Dishes: Adapting Object Counting Models to New Visual Domains and Ob- ject Types Without Forgetting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8070–8079, 2018.

Sanaz Mohammadjafari, Mucahit Cevik, Mathusan Thanabalasingam, and Ayse Basar. Using protop- net for interpretable alzheimer’s disease classifica- tion. In Canadian AI, 2021.

Abdurahman Ali Mohammed, Catherine Fonder, Ying Wei, Wallapak Tavanapong, Donald S. Sak- aguchi, Surya K. Mallapragada, and Qi Li. Cellfm- count: A fluorescence microscopy dataset, bench- mark, and methods for cell counting. In 2025 IEEE International Conference on Data Mining (ICDM), 2025a. To appear.

Abdurahman Ali Mohammed, Wallapak Ta- vanapong, Catherine Fonder, and Donald Sak- aguchi. Countxplain: Interpretable cell counting with prototype-based density map estimation. In Medical Imaging with Deep Learning, 2025b.

Christoph Molnar. uation of Interpretability.

Chapter 3.4 Eval-

2 edition,

2022.

[URL https://christophm.](https://christophm.github.io/interpretable-ml-book/evaluation-of-interpretability.html)

github.io/interpretable-ml-book/ evaluation-of-interpretability.html. [URL 🔗](https://christophm.github.io/interpretable-ml-book/evaluation-of-interpretability.html)


## Proto4DME

IEEE International Conference on Computer Vi- sion Workshops, pages 18–26, 2017.

Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine In- telligence 2019 1:5, 1:206–215, 5 2019. ISSN 2522- 5839. doi: 10.1038/s42256-019-0048-x.

Dawid Rymarczyk,  Lukasz Struski, Jacek Tabor, and Bartosz Zieli´nski. Protopshare: Prototypical parts sharing for similarity discovery in interpretable im- age classification. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, page 1420–1430, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383325.

Ville Satopaa, Jeannie Albrecht, David Irwin, and Barath Raghavan. Finding a” kneedle” in a haystack: Detecting knee points in system behav- ior. In 2011 31st international conference on dis- tributed computing systems workshops, pages 166– 171. IEEE, 2011.

Ramprasaath R. Selvaraju, Michael Cogswell, Ab- hishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual Explana- tions from Deep Networks via Gradient-Based Lo- calization. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 618–626, Oc- tober 2017. ISSN: 2380-7504.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In International conference on machine learning, pages 3145–3153. PMlR, 2017.

Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Im- age Recognition, April 2015. arXiv:1409.1556 [cs].

Gurmail Singh and Kin-Choong Yow. An inter- pretable deep learning model for covid-19 detec- tion with chest x-ray images. Ieee Access, 9:85198– 85208, 2021a.

Gurmail Singh and Kin-Choong Yow. These do not look like those: An interpretable deep learn- ing model for image recognition. IEEE Access, 9: 41482–41493, 2021b.

Alan Q Wang, Batuhan K Karaman, Heejong Kim, Jacob Rosenthal, Rachit Saluja, Sean I Young, and Mert R Sabuncu. A framework for interpretability in machine learning for medical imaging. IEEE Access, 12:53277–53292, 2024.

Boyu Wang, Huidong Liu, Dimitris Samaras, and Minh Hoai Nguyen. Distribution matching for crowd counting. Advances in neural information processing systems, 33:1595–1607, 2020.

Yuanpu Xie, Fuyong Xing, Xiaoshuang Shi, Xiangfei Kong, Hai Su, and Lin Yang. Efficient and Ro- bust Cell Detection: A Structured Regression Ap- proach. Medical image analysis, 44:245–254, Febru- ary 2018. ISSN 1361-8415. doi: 10.1016/j.media. 2017.07.003. URL https://www.ncbi.nlm.nih. gov/pmc/articles/PMC6051760/. [URL 🔗](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6051760/)

Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, and Yi Ma. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 589–597, June 2016. ISSN: 1063-6919.

Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning Deep Features for Discriminative Localization. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2921–2929. IEEE Computer Society, June 2016. ISBN 9781467388511.


## Appendix A. OT ablation: qualitative results

Qualitative analysis (Fig. 5). Figure 5 shows the top-K Proto4DME similarity maps for a rep- resentative MBM test image produced by a model trained without the OT balanced assignment loss. Al- though several prototypes fire close to dark, nucleus- like structures, the responses are largely diffuse and highly overlapping across prototypes. Many heatmaps highlight similar locations and extend broadly over tissue regions rather than concentrating tightly on cell instances. This redundancy suggests limited prototype specialization and weak pressure to provide complementary contributions. In the absence of OT, prototypes can instead gravitate toward ubiq- uitous textures and staining micro-patterns that are easy to match throughout the image, which qualita- tively manifests as widespread hotspots and reduced foreground–background selectivity. These visual pat- terns are consistent with our distance-based diag- nostics: without OT, the average minimum proto- type distance is smaller on background than on fore- ground (negative ∆dmin), indicating that prototypes are, on balance, better matched to background re- [URL 🔗](#page-0)

gions. Overall, Appendix Fig.

evidence that OT is important for promoting fore- ground coverage and discouraging prototype drift to- ward background cues.

[5](#page-0)

provides qualitative

## Appendix B. Prototype contribution breakdown example

Prototype contributions toward the predicted count per image. For each image, Proto4DME pro- duces a faithful breakdown of the predicted count into non-negative per-prototype terms. Concretely, the predicted count decomposes as ˆn = P k ck, where ck = P i,j wkEk(i, j) ≥ 0, Ek is the prototype acti- vation map, and wk ≥ 0 is the non-negative density- head coefficient (Eq. 9). Table 5 shows an example local explanation after pruning. All remaining proto- types are included, and their contributions sum ex- actly to the predicted count ˆn. In this example, the top three prototypes account for 65.5% of ˆn, provid- ing a compact summary of the main drivers of the prediction. Because contributions are strictly addi- tive and non-negative, prototype heatmaps can be interpreted directly as attributions for density, with- [URL 🔗](#page-0)

out cancellation effects.

*Table 5: Example local explanation for a single im-*

*age from the DCC dataset after prun- ing. Proto4DME decomposes the predicted count ˆn into non-negative per-prototype contributions ck. We report all remain- ing prototypes (post-pruning), and contri- butions sum exactly to ˆn.*

| Proto | gk | ck | % of ˆn |
| --- | --- | --- | --- |
| p1 | 0.967 |   | 3.372 23.16% |
| p4 | 0.575 |   | 3.269 22.45% |
| p3 | 0.952 |   | 2.893 19.88% |
| p0 | 0.846 |   | 2.619 17.99% |
| p2 | 0.915 |   | 2.405 16.52% |
| Top-3 total |   |   | 9.534 65.49% |
| All-prototypes total |   |   | 14.557 100% |
| Predicted count ˆn 14.557 100% |   |   |   |

*Table 6: Example local explanation for a single im-*

*age from the ADI dataset after prun- ing. Proto4DME decomposes the predicted count ˆn into non-negative per-prototype contributions ck. We report all remain- ing prototypes (post-pruning), and contri- butions sum exactly to ˆn.*

| Proto | gk | ck | % of ˆn |
| --- | --- | --- | --- |
| p5 | 0.989 |   | 33.617 16.08% |
| p4 | 0.961 |   | 26.310 12.58% |
| p0 | 0.952 |   | 24.346 11.64% |
| p6 | 0.956 |   | 23.503 11.24% |
| p2 | 0.933 |   | 22.771 10.89% |
| p7 | 0.951 |   | 22.399 10.71% |
| p8 | 0.919 |   | 20.586 9.85% |
| p1 | 0.936 |   | 18.113 8.66% |
| p3 | 0.946 |   | 17.434 8.34% |
| Top-3 total |   |   | 84.273 40.80% |
| All-prototypes total |   |   | 209.079 100% |
| Predicted count ˆn 209.079 100% |   |   |   |


## Proto4DME

*Figure 5: Prototype similarity maps obtained from a model that was trained without OT loss*

*Table 7: Example local explanation for a single im-*

*age from the MBM dataset after prun- ing. Proto4DME decomposes the predicted count ˆn into non-negative per-prototype contributions ck. We report all remain- ing prototypes (post-pruning), and contri- butions sum exactly to ˆn.*

| Proto | gk | ck | % of ˆn |
| --- | --- | --- | --- |
| p3 | 0.640 |   | 96.842 57.85% |
| p4 | 0.894 |   | 31.455 18.79% |
| p1 | 0.892 |   | 31.200 18.64% |
| p2 | 0.835 |   | 7.565 4.52% |
| p0 | 0.772 |   | 0.349 0.21% |
| Top-3 total |   |   | 159.497 95.28% |
| All-prototypes total |   |   | 167.411 100% |
| Predicted count ˆn 167.411 100% |   |   |   |

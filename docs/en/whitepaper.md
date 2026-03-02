
# LattiAI: A Development Platform for Privacy-Preserving AI Model Inference

## Model Secure Inference Overview

### Background and Need for Model Secure Inference

In recent years, the rapid development of artificial intelligence technology has been profoundly transforming how industries operate. In particular, the emergence of large language models such as ChatGPT, DeepSeek, Gemini, and Claude has marked breakthrough progress in AI capabilities for semantic understanding and knowledge reasoning, opening up possibilities for broader application scenarios.

As AI capabilities advance, cloud-based model inference services (Prediction as a Service, PaaS) have become the mainstream deployment approach. In this model, the AI model service provider and the client making inference requests are two distinct parties, creating risks of data privacy breaches during inference services. Information leakage in model inference involves multiple dimensions, with the most urgent and direct risk being the exposure of data from both participating parties. Generally speaking, model service providers need to protect the structure and parameters of their AI models, as this information corresponds to the model's intellectual property and training data privacy. At the same time, clients need to protect their request information and inference results, since input prompts, images, videos, audio, and other data often contain sensitive personal information. Without privacy-preserving computation technologies for Model Secure Inference, inference computation is typically performed by the service provider, allowing them direct access to the client's input data, intermediate computation results, and inference results, leading to leakage of the client's sensitive information. Conversely, if inference is performed by the client, the service provider must send model parameters to the client, resulting in leakage of model intellectual property. Therefore, these security requirements cannot be simultaneously satisfied, and the lack of privacy protection has become a major bottleneck preventing large-scale AI deployment in sensitive domains.

Model Secure Inference technology has emerged to address this challenge. Its goal is to achieve complete data privacy protection while ensuring computational correctness through cryptographic or hardware security techniques, solving the problem of direct data leakage during model inference. Under such security frameworks, clients can obtain correct model inference results without gaining any information about the model beyond the results themselves, while service providers cannot obtain any information about the requests, thereby satisfying the privacy protection needs of both parties.

Current mainstream technical approaches for Model Secure Inference include the following categories:

1. **Fully Homomorphic Encryption (FHE)**: Proposed by Gentry in 2009 ^[C. Gentry, "Fully Homomorphic Encryption Using Ideal Lattices," in Proceedings of the 41st annual ACM symposium on Symposium on theory of computing - STOC '09, Bethesda, MD, USA: ACM Press, 2009, p. 169. doi: 10.1145/1536414.1536440.], FHE allows computations to be performed directly on encrypted data, with decrypted results equivalent to computations on plaintext data. Current mainstream FHE schemes include BFV^[J. Fan and F. Vercauteren, "Somewhat Practical Fully Homomorphic Encryption," p. 19.], BGV^[Z. Brakerski, C. Gentry, and V. Vaikuntanathan, "(Leveled) Fully Homomorphic Encryption without Bootstrapping," p. 35.], CKKS^[Cheon, J.H., Kim, A., Kim, M., Song, Y. (2017). Homomorphic Encryption for Arithmetic of Approximate Numbers. In: Takagi, T., Peyrin, T. (eds) Advances in Cryptology – ASIACRYPT 2017. ASIACRYPT 2017. Lecture Notes in Computer Science, vol 10624. Springer, Cham. https://doi.org/10.1007/978-3-319-70694-8_15], and TFHE^[I. Chillotti, N. Gama, M. Georgieva, and M. Izabachène, "TFHE: Fast Fully Homomorphic Encryption Over the Torus," J Cryptol, vol. 33, no. 1, pp. 34–91, Jan. 2020, doi: 10.1007/s00145-019-09319-x.]. Among these, the CKKS scheme is most widely applied in deep learning inference due to its native support for floating-point arithmetic. FHE-based approaches use a non-interactive inference workflow: the client encrypts input data and sends it to the server, the server completes all model computations on the encrypted data, and finally returns the encrypted results to the client for decryption. FHE technology can achieve complete data privacy protection without any online interaction, though it incurs relatively high computational overhead. Additionally, FHE-based Model Secure Inference requires replacing nonlinear layers in AI models (such as ReLU activation functions) with polynomial approximations.

2. **Multi-Party Computation (MPC)**: Allows multiple parties to collaboratively compute a predetermined function without revealing their private data. This approach uses an interactive inference workflow: the client and server split data through Secret Sharing, and both parties collaboratively complete computation for each layer of the model. For linear layers (such as convolution and fully connected layers), this is typically implemented using homomorphic encryption or Oblivious Transfer (OT)^[Ishai, Y., Kilian, J., Nissim, K., & Petrank, E. (2003). Extending oblivious transfers efficiently. In Annual International Cryptology Conference (pp. 145-161). Berlin, Heidelberg: Springer Berlin Heidelberg.] techniques. For nonlinear layers (such as ReLU and MaxPool), Garbled Circuits (GC)^[Yao, A. C. C. (1986). How to generate and exchange secrets extended abstract. In 27th FOCS (pp. 162-167).] or OT protocols are used. MPC technology can achieve high-precision computation for nonlinear layers, but multiple rounds of interactive communication introduce significant communication overhead.

3. **Confidential Computing**: Based on hardware security technologies such as Trusted Execution Environments (TEE), this approach executes sensitive computations in isolated secure regions. This technology relies on hardware root of trust.

Additionally, Model Secure Inference solutions can be combined with other privacy-enhancing technologies:

- **Differential Privacy (DP)**: Protects training data privacy and prevents membership inference attacks by adding carefully designed noise to model outputs.

- **Federated Learning (FL)**: Allows multiple parties to collaboratively train models without sharing raw data, which can be combined with secure inference techniques to achieve end-to-end privacy protection.

It should be noted that while FHE or MPC Model Secure Inference technologies can theoretically provide rigorous privacy protection, they often introduce orders-of-magnitude performance overhead compared to plaintext inference, presenting challenges in efficiency and scalability. Therefore, how to optimize computation time or communication overhead while ensuring security is one of the key issues for practical deployment of Model Secure Inference technology.

Furthermore, how to automatically and conveniently convert AI models into versions that support Model Secure Inference is also a critical issue. This conversion process may require sacrificing some model accuracy, and different secure inference models have significant differences in inference time or communication overhead, making it another core challenge to obtain relatively optimal secure inference models.

This product provides innovative solutions to address these challenges.

### CKKS Homomorphic Encryption

CKKS is a homomorphic encryption scheme that supports fixed-point arithmetic operations on encrypted data. It is widely applied in secure AI model inference due to its support for floating-point numbers and SIMD (Single Instruction, Multiple Data) characteristics. In this scheme, vector data $\mathbf{m} \in \mathbb{R}^{N/2}$ (or $\mathbb{C}^{N/2}$) is first encoded as a plaintext polynomial in the cyclotomic ring $\mathcal{R}_Q = \mathbb{Z}_Q[X]/(X^N + 1)$ (where $N$ is a power of 2), then encrypted into ciphertext by introducing noise.

The key feature of CKKS is **Single Instruction, Multiple Data (SIMD)**: a single homomorphic operation (such as addition or multiplication) can complete element-wise operations on all $N/2$ elements of the encrypted vector in one step. Each position in the encrypted vector is called a **slot**.

#### Core Homomorphic Operations in CKKS

Where $[\mathbf{u}]$ represents the ciphertext of encrypted vector $\mathbf{u}$:

**1. Homomorphic Addition and Multiplication**

- Homomorphic addition and multiplication can be performed between two ciphertexts or between a ciphertext and a plaintext
- Both addition and multiplication are executed element-wise

**2. Homomorphic Rotation**

- Cyclically shifts slots in the ciphertext
- When rotating by $r$ positions, shifts left when $r>0$ and shifts right when $r<0$

#### Multiplicative Depth and Ciphertext Refreshing

A major limitation of homomorphic encryption is the restricted depth of multiplication operations. We use the ciphertext's **level** to represent its available number of multiplications or maximum multiplicative depth:

- Each homomorphic multiplication operation decreases the ciphertext's level by 1
- When the ciphertext's level is exhausted (level=0), bootstrapping^[Cheon, J.H., Han, K., Kim, A., Kim, M., Song, Y. (2018). Bootstrapping for Approximate Homomorphic Encryption. In: Nielsen, J., Rijmen, V. (eds) Advances in Cryptology – EUROCRYPT 2018. EUROCRYPT 2018. Lecture Notes in Computer Science, vol 10820. Springer, Cham. https://doi.org/10.1007/978-3-319-78381-9_14] operations must be performed to refresh the ciphertext's available multiplicative depth

### Overall Workflow for Model Secure Inference

Based on FHE technology's capability to perform computations on encrypted data, we can construct a general-purpose secure AI model inference solution. The overall workflow is illustrated in the following diagram:

<img src="../images/workflow/inference_simple_flow_en.png" alt="Model Secure Inference Workflow" title="Model Secure Inference Workflow"/>

The client only sends encrypted data to the service provider, so the service provider cannot obtain any information about the client's data. The service provider only sends the encrypted model inference results back to the client, who after decryption can only obtain the inference results without gaining any additional information. In this way, this encrypted computation workflow clearly satisfies the security objectives of both parties, achieving privacy protection for data from both sides.

Because the CKKS fully homomorphic encryption algorithm natively supports floating-point computation, efficient vectorized parallel computation, and has a mature bootstrapping scheme, our product primarily uses the CKKS algorithm to perform encrypted inference for deep neural networks.

A linear layer in a deep neural network (including various types of convolutional layers, fully connected layers, average pooling layers, etc.) is a linear transformation of input features. Its FHE-based computation generally consumes one or two multiplicative depths, with specific high-performance implementation methods to be introduced in subsequent sections. The following diagram shows an example of the basic computation pattern for a linear layer:

<img src="../images/workflow/layer0_en.png" title="layer0" width="45%"/>

Where $L$ represents the multiplicative depth that the ciphertext can support.

A nonlinear layer in a neural network (including various activation function layers, max pooling layers, etc.) generally has no model parameters. Activation function layers perform element-wise nonlinear function transformations on input features, while max pooling layers perform nonlinear computations between different points of input features. Their FHE-based computation generally consumes one or more multiplicative depths, with specific high-performance implementation methods to be introduced in subsequent sections. The following diagram shows an example of the basic computation pattern for a nonlinear layer:

<img src="../images/workflow/layer1_en.png" title="layer1" width="20%"/>

For a multi-layer deep neural network, we need to refresh the ciphertext where the available multiplicative depth is exhausted in order to restore the available multiplicative depth of feature ciphertexts and continue computation for subsequent layers. Our product uses the FHE-Bootstrapping mode for ciphertext refreshing, using the CKKS algorithm's bootstrapping operator to perform re-encryption directly on the ciphertext state, with the workflow shown in the following diagram:

<img src="../images/workflow/layer2_en.png" title="layer2" width="20%"/>

Within this general framework, we need to consider the characteristics of FHE algorithms and maximize the overall performance of model inference computation without compromising security or model inference accuracy. Next, we will introduce efficient FHE-based computation methods for several of the most typical neural network layers.

## Operator Solutions for Deep Neural Network Secure Inference

### Data Packing Schemes

Each CKKS ciphertext can pack $N/2$ data units (such as image pixel values). Different packing schemes affect the number of ciphertexts required to pack all data and the ciphertext computation logic, which in turn impacts the computational and communication overhead of secure inference. From the perspective of image channel dimensions, traditional packing schemes mainly include the following three types. The diagram below shows three packing methods using an example image with 4 channels, each channel having spatial dimensions of $2\times 2$, where $N/2=16$:

![](../images/algorithms/packing.jpg)

1. **Continuous Packing**: Image pixels are stored sequentially and continuously in ciphertext slots in row-major order. Depending on the ciphertext slot capacity, a single CKKS ciphertext can contain one or multiple complete channels (as shown in packing method 1 above).

2. **Multiplexed Packing**: Within a single CKKS ciphertext, slots alternately store pixel data from multiple channels, such that pixels from each channel are distributed across the entire ciphertext slot sequence at certain intervals (as shown in packing method 2 above).

3. **Gap Packing**: A single CKKS ciphertext stores one image channel, but elements within the channel are not stored continuously in the ciphertext slots; instead, they are stored with certain gaps that are filled with zero values (as shown in packing method 3 above).

However, these traditional packing schemes have limitations when handling high-resolution images (where the number of pixels per channel exceeds the CKKS ciphertext slot capacity $N/2$). For example, they can cause difficulties in convolution computation at channel split boundaries or data packing format compatibility issues when connecting different encrypted operators.

#### Generalized Interleaved Packing Scheme

To systematically support encrypted inference for images of virtually arbitrary resolution, including high-resolution images, we propose the **Generalized Interleaved Packing (GIP)** scheme and implement end-to-end encrypted inference through correspondingly designed GIP-preserving encrypted operators[^peregrine].

##### Channel Packing Factor

The Generalized Interleaved Packing scheme introduces the concept of **channel packing factor** $g$:

$$g = \frac{H}{\hat{H}}$$

Where $H \times H$ represents the image spatial resolution (assumed to be square), and $\hat{H}^2$ represents the base packing size that does not exceed the CKKS ciphertext slot capacity.

Let $g_i$ denote the channel packing factor for the input of the $i$-th computational layer in the deep neural network ($i=0,1,\ldots,L$), and $g_{L+1}$ denote the channel packing factor for the final output of the deep neural network. The calculation rules for channel packing factors are:

- If layer $i$ is a downsampling operator (such as convolution or pooling with stride $s_i$): $g_{i+1} = g_i / s_i$
- If layer $i$ is an upsampling operator (such as transposed convolution or nearest neighbor upsampling with stride $\hat{s}_i$): $g_{i+1} = g_i \cdot \hat{s}_i$
- If layer $i$ is a resolution-preserving operator (such as convolution with stride 1, activation functions, batch normalization, etc.): $g_{i+1} = g_i$

##### Forms of Generalized Interleaved Packing

Based on the value of the channel packing factor $g$, the Generalized Interleaved Packing scheme adaptively adopts different packing forms:

- **$g > 1$**: The number of pixels per channel exceeds the base packing size. Each channel is periodically sampled in row and column directions at intervals of $g$, decomposed into $g^2$ interleaved sub-channels, with each sub-channel independently packed into one ciphertext.
- **$g < 1$**: The number of pixels per channel is less than the base packing size. This corresponds to multiplexed packing, where the slots of a single ciphertext are divided into $N/(2 \hat{H}^2)$ consecutive blocks, with each block packing $1/g^2$ channels in an interleaved manner.
- **$g = 1$**: The number of pixels per channel exactly equals the base packing size. This degenerates to continuous packing, where the slots of a single ciphertext pack $N/(2 \hat{H}^2)$ input channels.

#### Encrypted Computation Implementation for Different Neural Network Layers

To efficiently support encrypted computation of floating-point numbers, we use the CKKS homomorphic encryption algorithm to perform encrypted inference computations for models. In the CKKS homomorphic encryption algorithm, a ciphertext encrypts (packs) a plaintext vector containing $N/2$ floating-point numbers. Homomorphic addition or multiplication operations performed on ciphertexts correspond to parallel addition or multiplication computations of $N/2$ floating-point numbers—this is the SIMD (single instruction, multiple data) characteristic of the CKKS algorithm. To fully leverage the parallel computing power of SIMD, it is necessary to specifically design encrypted computation methods for fully connected layers, convolutional layers, and other operators.

### Convolutional Layers

Encrypted computation for convolutional layers is implemented through a series of homomorphic rotation, multiplication, and addition operations. Based on the Generalized Interleaved Packing scheme, our designed encrypted convolution operators can preserve the packing structure, meaning both input and output maintain the Generalized Interleaved Packing format. Implementation methods differ based on the different values of the channel packing factor $g$.

#### Basic Convolution Implementation ($g=1$)

##### Stride $s$, $s=1$

When $g=1$, the packing method is continuous packing. The encrypted convolution implementation method for this scenario comes from Gazelle^[Juvekar, C., Vaikuntanathan, V., & Chandrakasan, A. (2018). GAZELLE: A low latency framework for secure neural network inference. In 27th USENIX security symposium (USENIX security 18) (pp. 1651-1669). https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar].

**Core Idea**: Move input elements to their contribution positions for output through homomorphic rotation, then perform homomorphic multiplication and addition to complete the convolution computation.

**Implementation Principle**:

Let the center coordinates of the convolution kernel be $(c_x, c_y)$. In standard convolution, the value at output position $(x,y)$ is obtained from the dot product of all kernel elements with corresponding input elements when the kernel center is aligned with that position. Specifically:

- The offset of kernel element $f_{(i,j)}$ relative to the center is $(i-c_x, j-c_y)$
- This element is multiplied by the element at input position $(x+(i-c_x), y+(j-c_y))$, with the result contributing to output position $(x,y)$
- **Key Property**: The relative position between the input element and output element participating in multiplication is consistent with the relative position between the multiplied kernel element and the kernel center

Gazelle's encrypted convolution leverages this property by performing the following operations for each kernel element $f_{(i,j)}$:

1. **Rotation Alignment**: Rotate the entire input ciphertext by $(i-c_x) * W + j-c_y$ steps, where $W$ is the row width of the input image, moving input elements to their contribution positions for output. Positive steps correspond to left rotation, negative steps to right rotation
2. **Element-wise Multiplication**: Multiply the rotated ciphertext by the weight plaintext encoding $f_{(i,j)}$
3. **Accumulative Summation**: Sum all multiplication results of rotated ciphertexts and kernel element plaintexts to obtain the output ciphertext

**Detailed Steps**:

**Step 1: Ciphertext Rotation and Weight Encoding**

For each kernel element $f_{(i,j)}$, perform:

- **Calculate Rotation Steps**: Based on the element's offset relative to the center $(i-c_x, j-c_y)$, calculate the rotation steps as $(i-c_x) \times W + j-c_y$, where $W$ is the row width of the input image

- **Rotate Input Ciphertext**: Perform homomorphic rotation on the input ciphertext by the calculated steps

- **Encode Weight Plaintext**: Encode the plaintext polynomial corresponding to $f_{(i,j)}$, with slot element assignment rules as follows:
  - If the slot is reached by rotation from the region scanned by $f_{(i,j)}$, store $f_{(i,j)}$
  - Otherwise, store zero value

**Step 2: Homomorphic Multiply-Add**

- Multiply all rotated ciphertexts with their corresponding weight plaintexts, then accumulate all intermediate multiplication results to obtain the convolution output ciphertext

To more intuitively understand this process, the following diagram shows an example for a single input-single output channel scenario. In this example, the input image size is $3 \times 3$, the convolution kernel size is $3 \times 3$, the convolution stride is $1 \times 1$, and padding is $1 \times 1$. The image width $W=3$, and the kernel center coordinates $(c_x, c_y)$ are $(1,1)$. By performing a series of homomorphic rotations, homomorphic multiplications, and homomorphic additions on the ciphertext $c_{in}$ of the encrypted input image, the encrypted computation of basic convolution can be implemented.

Taking the rotation steps corresponding to different kernel coefficients as examples:

- The rotation steps for $f_{(0,0)}$ are $(0-1) \times 3 + (0-1) = -4$, corresponding to a right shift of 4 steps: operation $\mathrm{Rot}_{-4}$
- The rotation steps for $f_{(2,2)}$ are $(2-1) \times 3 + (2-1) = 4$, corresponding to a left shift of 4 steps: operation $\mathrm{Rot}_{4}$

<img src="../images/algorithms/gazelle_convolution.png" title="Gazelle_convolution" width="90%"/>

Supplementary Note: The above two-dimensional diagram is intended to visually illustrate the encrypted computation process of basic convolution. In actual encrypted computation, image pixels and kernel coefficients need to be packed into one-dimensional slots of ciphertext and plaintext respectively in row-major order.

##### Stride $s$, $s>1$

**Core Idea**: First execute encrypted convolution with stride 1 and preserve output positions for stride $s$ through masking, then transform gap packing into multiplexed packing.

**Implementation Principle**:

Convolution output with stride $s$ is equivalent to sampling the result of stride-1 convolution every $s$ positions in row and column directions. Encrypted computation can be implemented in two stages:

1. **Masked Convolution**: Execute encrypted convolution with stride 1, inserting a sampling mask during weight encoding to make non-output positions zero

- Specific approach: For each weight plaintext, if the slot index is not a multiple of $s$, set that slot value to 0
- Output is in gap packing form, with valid values separated by intervals of $s$

2. **Packing Format Conversion**: Transform gap packing into multiplexed packing

- According to the channel packing factor definition, the output channel packing factor is $g_{out} = 1/s$
- Through homomorphic rotation and summation operations, fill zero positions in intervals with data from other channels
- Finally, each ciphertext packs $1/s^2 * N/(2\hat{H}^2)$ output channels, forming a multiplexed packing format

#### Low-Resolution Image Convolution ($g<1$)

When the input channel packing factor $g<1$, the input ciphertext uses multiplexed packing format, where the slots of each ciphertext are divided into $N/(2\hat{H}^2)$ consecutive blocks, with each block packing $1/g^2$ input channels in an interleaved manner. The encrypted convolution implementation method for this scenario comes from the paper that proposed multiplexed packing^[Lee, E., Lee, J. W., Lee, J., Kim, Y. S., Kim, Y., No, J. S., & Choi, W. (2022). Low-complexity deep convolutional neural networks on fully homomorphic encryption using multiplexed parallel convolutions. In International Conference on Machine Learning (pp. 12403-12422). PMLR. https://eprint.iacr.org/2021/1688].

In the multiplexed packing format, data from multiple channels is interleaved within the ciphertext, which provides natural advantages for parallel computation of channel convolutions.

**Core Idea**: Leverage the characteristics of multiplexed packing to compute convolutions of multiple input channels with their corresponding kernels in parallel within a single ciphertext, then aggregate results through homomorphic rotation, summation, and masking operations.

Based on the above idea, the implementation of encrypted convolution can be divided into the following steps:

**Step 1: Parallel Convolution Computation**

For each input channel, perform operations similar to basic convolution:

- For each element of the convolution kernel, rotate the input ciphertext and multiply with the encoded weight plaintext
- Accumulate all products to obtain a ciphertext containing convolution results from multiple channels

**Step 2: Result Aggregation**

Aggregate each channel result interleaved in the ciphertext to designated positions:

- Through homomorphic rotation and summation, accumulate the convolution results of all input channels with their corresponding kernels to the 0th channel position of the 0th block in the intermediate ciphertext, corresponding to the valid value of one output channel, with other positions being invalid values

**Step 3: Output Packing**

Repack each output channel into multiplexed format:

- Apply a mask to each output channel to extract valid values for that channel (in gap packing format)
- Rotate to target positions and accumulate to form output ciphertext in multiplexed packing

To further improve computational efficiency, our product performs innovative optimizations on top of the basic implementation scheme.

**Product Optimization Strategy**:

In the basic implementation, the intermediate ciphertext obtained from steps 1 and 2 contains valid values for only one output channel at the 0th channel position of the 0th block, with all other positions being invalid values. To improve efficiency, we advance block-direction rotation to step 1, such that different block positions are multiplied by kernels corresponding to different output channels, allowing parallel computation of $N/(2\hat{H}^2)$ output channels in a single intermediate ciphertext, stored respectively at the 0th channel position of each block. This optimization significantly reduces the number of ciphertext rotations and lowers inference time overhead.

#### High-Resolution Image Convolution ($g>1$)

When $g>1$, the number of pixels per channel exceeds the capacity of a single ciphertext slot, and each channel of the image is decomposed and packed into different ciphertexts in an interleaved manner according to the channel packing factor. This is the core innovation of our proposed Generalized Interleaved Packing scheme, solving the problem that traditional schemes cannot effectively handle high-resolution images. For the case of $g>1$, we have designed a specialized encrypted convolution algorithm that both maintains the structural characteristics of Generalized Interleaved Packing and efficiently completes convolution computation.

##### Core Idea

The basic approach of the proposed algorithm is to fully leverage the interleaved structure characteristics of sub-channels. Specifically, it receives input images in Generalized Interleaved Packing format, where each channel is decomposed into multiple interleaved sub-channels, performs basic convolution operations on each input sub-channel, accumulates intermediate results, and obtains output images in Generalized Interleaved Packing format.

##### Implementation Principle:

For encrypted convolution with stride $s_i$, the input image channel packing factor is $g_i$, and the output image channel packing factor is $g_{i+1} = g_i / s_i$. According to the Generalized Interleaved Packing scheme, each channel of the input image is decomposed into $g_i^2$ interleaved sub-channels, packed into $g_i^2$ ciphertexts; correspondingly, each channel of the output image is decomposed into $g_{i+1}^2$ interleaved sub-channels, packed into $g_{i+1}^2$ ciphertexts.

To understand how to compute these output sub-channels, we observe a key property: each output sub-channel can be obtained by performing convolution with stride $g_i$ on the input image. In convolution with stride $g_i$, the input elements scanned by each kernel coefficient correspond exactly to one input sub-channel; furthermore, coefficients in the kernel with row-column intervals of $g_i$ scan the same input sub-channel. Based on this observation, the ciphertext packing each output sub-channel can be obtained by performing basic convolution with stride 1 on the ciphertexts packing input sub-channels.

##### Detailed Steps

Based on the above principles, we can decompose the computation process of encrypted convolution into the following three main steps:

1. Starting from different positions of the convolution kernel, sample at row-column intervals of $g_i$ to obtain several disjoint sub-kernels.
2. Perform basic encrypted convolution with stride 1 on the relevant input sub-channel ciphertexts and their corresponding multiplied sub-kernels
3. Accumulate the multiplication results of all input sub-channels with sub-kernel coefficients to obtain the Generalized Interleaved Packing of the output image

Taking the single input-output channel scenario in the following diagram as an example, both input and output have a channel packing factor of 2, spatial resolution of $8\times 8$, kernel size of $3\times 3$, padding size of $1\times 1$, and convolution stride of $1\times 1$:

<img src="../images/algorithms/large-size-convolution-en.png" title="large-size convolution" width="90%"/>

As shown in the diagram, the four sub-channels obtained by decomposing the output channel based on the Generalized Interleaved Packing format can all be obtained by summing the results of performing basic convolution with stride 1 on input sub-channels and corresponding sub-kernels. This encrypted convolution maintains the Generalized Interleaved Packing format, meaning both input and output are encrypted images in Generalized Interleaved Packing.

#### Depthwise Convolutional Layers

Depthwise Convolution is a special convolutional operator widely used in lightweight neural network architectures such as MobileNet. Unlike standard convolution, depthwise convolution separates spatial convolution from linear combinations across channels, significantly reducing the number of parameters and computational cost. In encrypted inference scenarios, this separation characteristic provides us with implementation convenience.

**Core Idea**: Each input channel independently performs convolution operations to obtain one output channel, without cross-channel summation.

**Implementation Principle**:

To more clearly understand the implementation of encrypted depthwise convolution, we first compare the main differences between depthwise convolution and standard convolution:

- Standard convolution: Each output channel is obtained by accumulating the convolution results of all input channels with corresponding kernels
- Depthwise convolution: Each input channel independently corresponds to one output channel, with no cross-channel summation computation

**Detailed Steps**:

Based on this characteristic, the encrypted implementation process is as follows:

- For each input channel, perform homomorphic rotation, multiplication, and addition operations of basic convolution

The output of encrypted depthwise convolution maintains the Generalized Interleaved Packing format.

#### Transposed Convolutional Layers

Transposed Convolution (also known as Deconvolution, using PyTorch's torch.nn.ConvTranspose2d as the definition standard) is a commonly used upsampling operation in deep learning, widely applied in tasks such as image segmentation and generative adversarial networks that need to restore spatial resolution. In plaintext computation, transposed convolution operation is equivalent to first performing zero-inserted upsampling on all input channels, then performing normal convolution computation on the zero-inserted upsampled image based on the transpose of the transposed convolution weights. Based on this equivalence relationship, we can fully utilize existing encrypted convolution operators to implement encrypted transposed convolution.

Based on the above idea, encrypted transposed convolution can be decomposed into the following three steps:

**Step 1: Zero-Inserted Upsampling**

Perform zero-inserted upsampling on each channel of the input image, such that the element spacing equals the transposed convolution stride $\hat{s}_i$, with the gaps filled with zero values.

Implementation methods differ based on the input channel packing factor $g_i$:

- **$g_i < 1$**: Combine ciphertext rotation and masking operations to generate zero-inserted upsampling results in the encrypted state

- **$g_i \geq 1$**: According to the Generalized Interleaved Packing scheme, each input channel is decomposed into $g_i^2$ sub-channels and independently packed into different ciphertexts. After zero-insertion, the upsampled channel is decomposed into $g_i^2 \cdot \hat{s}_i^2$ sub-channels, of which $g_i^2$ correspond to original input sub-channels, with the rest being all-zero sub-channels. Therefore, inserting $g_i^2 \cdot \hat{s}_i^2 - g_i^2$ all-zero ciphertexts can generate zero-inserted upsampling results in the encrypted state.

**Step 2: Weight Transpose Encoding**

Transpose the weights of the transposed convolution operator and encode their elements as plaintext polynomials.

**Step 3: Encrypted Convolution**

Perform corresponding encrypted convolution operations on the upsampled image to obtain transposed convolution results.

The output of encrypted transposed convolution maintains the Generalized Interleaved Packing format.

#### Nearest Neighbor Upsampling Layers

Nearest Neighbor Upsampling is another commonly used upsampling operation, widely used in object detection networks such as YOLO. In plaintext computation, nearest neighbor upsampling operation is equivalent to first performing zero-inserted upsampling on the input image, then filling zero-value positions with adjacent input pixel values, such that each input pixel is repeated $\hat{s}_i^2$ times, where $\hat{s}_i$ is the upsampling stride. This equivalence relationship provides a clear implementation path for its encrypted implementation.

Based on the above idea, encrypted nearest neighbor upsampling can be decomposed into the following two steps:

**Step 1: Zero-Inserted Upsampling**

Perform zero-inserted upsampling on each channel of the input image, such that the element spacing equals the upsampling stride $\hat{s}_i$. Implementation method is the same as Step 1 of transposed convolutional layers.

**Step 2: Pixel Replication**

Based on the zero-inserted upsampling results, perform several homomorphic rotation and summation operations to fill zero-value positions with adjacent input image elements, such that each pixel of the input image is repeated $\hat{s}_i^2$ times.

The output of encrypted nearest neighbor upsampling maintains the Generalized Interleaved Packing format.

### Batch Normalization Layers

Batch Normalization is a widely used normalization technique in deep neural networks, used to accelerate training convergence and improve model generalization capability. During the training phase, batch normalization layers need to compute the mean and variance of current batch data; during the inference phase, batch normalization layers use running statistics accumulated during the training process, at which point their computation process degenerates into a linear operator with completely fixed parameters independent of input data.

Specifically, the batch normalization operation during inference can be expressed as $y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$, where $\mu$ and $\sigma^2$ are the fixed running mean and variance, and $\gamma$ and $\beta$ are learnable scaling and shift parameters. This linear transformation can be equivalently rewritten in the form $y = w \cdot x + b$, where $w = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$ and $b = \beta - \frac{\gamma \cdot \mu}{\sqrt{\sigma^2 + \epsilon}}$.

Based on this property, to reduce ciphertext level consumption and improve computational efficiency, our product fuses batch normalization layer parameters into the previous convolutional layer. The specific approach is to merge the linear transformation parameters of batch normalization into the weights and biases of the convolutional layer, so that batch normalization operations do not need to be performed separately during encrypted inference, both saving one ciphertext multiplication operation and maintaining the correctness of computation results.

### Fully Connected Layers

Fully Connected Layers (Dense Layers) are fundamental components in neural networks, commonly used for classification heads or feature transformation, with the role of mapping input features to the output space. The plaintext computation of fully connected layers is essentially matrix-vector multiplication, with the computation formula:

$$
y_i = \sum_{j=0}^{n_{in}-1}w_{ij} x_j + b_i
$$

Where $x_j,\ j\in\{0,1,\ldots,n_{in}-1\}$ are fully connected layer input values, $y_i,\ i\in\{0,1,\ldots,n_o-1\}$ are fully connected layer output values, and $w_{ij}$ and $b_i$ are fully connected layer weight parameters. When $n_{in}=n_o=4$, the matrix form representation of fully connected layer plaintext computation is:

$$\left[ \begin{matrix} y_{0}\\y_{1}\\y_{2}\\y_{3} \end{matrix} \right] =\left[ \begin{matrix} w_{00} & w_{01} & w_{02} & w_{03}\\ w_{10} & w_{11} & w_{12} & w_{13}\\ w_{20} & w_{21} & w_{22} & w_{23}\\ w_{30} & w_{31} & w_{32} & w_{33} \end{matrix} \right] \cdot \left[ \begin{matrix} x_{0}\\x_{1}\\x_{2}\\x_{3} \end{matrix} \right] + \left[ \begin{matrix} b_{0}\\b_{1}\\b_{2}\\b_{3} \end{matrix} \right]
$$

In the encrypted scenario, directly implementing the above matrix-vector multiplication requires significant computational overhead. To efficiently utilize CKKS's SIMD parallel computing capability, special data packing and computation schemes need to be designed. Our product supports the encrypted computation scheme for fully connected layers from the existing Gazelle method, with two implementation approaches: basic diagonal packing and hybrid diagonal packing.

#### Basic Diagonal Packing Implementation

The diagonal packing scheme is an ingenious implementation method that can efficiently complete matrix-vector multiplication by reorganizing the weight matrix in diagonal directions and coordinating with ciphertext rotation operations.

**Core Idea**: Align elements at different positions of the input vector with diagonal elements of the weight matrix through homomorphic rotation, then perform homomorphic multiplication and addition to complete matrix-vector multiplication.

Taking the scenario of $n_{in}=n_o=4$ as an example, the implementation steps are explained in detail below:

**Step 1: Input Data Packing**

The input vector uses continuous packing, such as $[x_0, x_1, x_2, x_3]$

**Step 2: Weight Parameter Diagonal Packing**

The weight matrix is packed in diagonal directions. Each diagonal is packed as a plaintext vector:

- Main diagonal: $[w_{00}, w_{11}, w_{22}, w_{33}]$
- Sub-diagonal 1: $[w_{01}, w_{12}, w_{23}, w_{30}]$
- Sub-diagonal 2: $[w_{02}, w_{13}, w_{20}, w_{31}]$
- Sub-diagonal 3: $[w_{03}, w_{10}, w_{21}, w_{32}]$

**Step 3: Ciphertext Rotation**

Rotate the input ciphertext according to weight packing positions:

- Original ciphertext: $[x_0, x_1, x_2, x_3]$ (rotate 0 steps)
- Rotated ciphertext 1: $[x_1, x_2, x_3, x_0]$ (rotate 1 step)
- Rotated ciphertext 2: $[x_2, x_3, x_0, x_1]$ (rotate 2 steps)
- Rotated ciphertext 3: $[x_3, x_0, x_1, x_2]$ (rotate 3 steps)

**Step 4: Homomorphic Multiply-Add**

Multiply the rotated ciphertexts with their corresponding weight plaintexts, then accumulate all products to obtain the fully connected layer output ciphertext.

#### Hybrid Diagonal Packing Implementation

In practical applications, fully connected layers often have output dimensions much smaller than input dimensions (such as the last layer of classification tasks). For this scenario, the hybrid diagonal packing scheme provides a more efficient implementation method. The hybrid diagonal packing scheme extends the square matrix diagonal method to rectangular weight matrices, combining generalized diagonal packing with block summation to replace numerous input rotations with fewer output rotations.

### Activation Layers

Activation layers are indispensable components in deep neural networks, breaking through the limitations of linear superposition by introducing nonlinear transformations, enabling networks to fit complex functions. Activation layers apply the same nonlinear activation function to each element of input features, with common activation functions including ReLU, SiLU, Sigmoid, etc.

However, in encrypted inference scenarios, implementing activation layers faces significant challenges: these nonlinear activation functions typically cannot be precisely represented by finite-degree polynomials, while CKKS homomorphic encryption only natively supports addition and multiplication operations and cannot directly compute arbitrary nonlinear functions. This contradiction makes activation layers one of the key difficulties in encrypted inference.

To address this challenge, our product uses the **polynomial approximation-based homomorphic computation mode**: it approximates original activation functions with low-degree polynomials, implemented through homomorphic multiplication and addition operations in the encrypted state without any interactive communication. To compensate for accuracy loss from polynomial approximation, our product proposes innovative training strategies that can ensure the accuracy of replaced models is comparable to original models. This approach has high computational efficiency and no communication overhead, suitable for scenarios where the server handles all computational tasks.

### Pooling Layers

Pooling layers are important components in convolutional neural networks, used to reduce the spatial dimensions of feature maps, decrease the number of parameters and computational cost, while extracting more robust feature representations. Common pooling operations include two types: average pooling and max pooling, which have different characteristics and challenges in encrypted implementation.

#### Average Pooling Layers

Average pooling layers have natural implementation advantages in encrypted scenarios. From a mathematical perspective, the computation of average pooling layers is equivalent to depthwise convolution with specific kernel coefficients, where all kernel coefficients are $1/k^2$ ($k\times k$ is the kernel size). Based on this equivalence relationship, the implementation of encrypted average pooling operators based on the Generalized Interleaved Packing scheme is consistent with encrypted depthwise convolution operators, fully compatible with the Generalized Interleaved Packing format, and can be seamlessly integrated into the entire encrypted inference workflow.

#### Max Pooling Layers

Unlike average pooling, max pooling layers face greater challenges in encrypted implementation. Max pooling needs to select the maximum value from multiple input values, which involves comparison operations, and comparison operations cannot be efficiently implemented directly through CKKS homomorphic operations.

To address this challenge, our product uses **average pooling layer replacement**:

Replace max pooling layers with average pooling layers, leveraging the encrypted implementation advantages of average pooling. Since this replacement changes the computational logic of the network, fine-tuning or retraining of the replaced model is needed to compensate for accuracy loss from the difference between the two pooling methods, thereby maintaining original model accuracy. This approach requires no interactive communication and has higher computational efficiency, but may cause some model accuracy loss.

## Model Compilation and Retraining

In Model Secure Inference scenarios, users typically want to directly convert trained plaintext models into encrypted models that support secure inference without needing to understand complex cryptographic details. However, this conversion process faces numerous technical challenges: ciphertexts consume multiplicative depth after homomorphic multiplication operations, and when multiplicative depth is exhausted, it must be refreshed through Bootstrapping operations; nonlinear layers need to be replaced with polynomial approximations; different Model Secure Inference strategies lead to vastly different computational overhead.

To address these challenges and achieve a fully automated workflow for Model Secure Inference, our product provides an intelligent compiler tool. The core function of the compiler is to transform the computation graph of a plaintext model into the corresponding secure inference model computation graph according to the selected inference mode. Specifically, the compiler first transforms the layer-by-layer forward inference process of the model into a Directed Acyclic Graph (DAG), where the computational layers of the model serve as computational nodes in the graph, and the inputs and outputs of computational layers serve as data nodes in the graph. Then, the compiler inserts ciphertext refreshing nodes (Bootstrapping layers) into the original model's computation graph according to certain criteria, and appropriately processes nonlinear layers to generate multiple feasible secure inference computation graph schemes. Finally, through intelligent search algorithms, it selects the optimal scheme with minimal computational overhead from these feasible schemes, thereby ensuring the continuity and efficiency of Model Secure Inference.

### Model Structure Compilation

#### Compiler Supported Features

Our product compiler has powerful model conversion capabilities, able to automatically handle multiple network architectures and operator types, and supports flexible inference mode configuration.

##### Supported Encrypted Inference Modes

Our product compiler supports the following encrypted inference mode:

**Homomorphic Encryption + Bootstrapping**

This mode adopts a pure FHE scheme without requiring online interaction between client and server:

- Uses CKKS homomorphic encryption for encrypted computation
- Replaces all model's nonlinear layers with approximate polynomial functions
- Inserts Bootstrapping computation nodes at appropriate positions to refresh the available multiplicative depth of ciphertexts based on the multiplicative depth consumed by each computational layer of the model
- Uses intelligent search algorithms to find optimal Bootstrapping insertion positions, minimizing overall computational overhead while ensuring computational correctness

This mode is suitable for scenarios sensitive to communication latency or requiring offline inference.

##### Supported Network Architectures

The compiler supports transforming multiple mainstream convolutional neural network architectures into secure inference models, covering various application domains such as image classification, object detection, and image segmentation:

- Image Classification: ResNet, MobileNet, etc.
- Object Detection: YOLO series, etc.
- Image Segmentation: U-Net, etc.

##### Supported Operator Types

The compiler has implemented automated conversion for common neural network operators, including:

- Linear layers: convolution, depthwise convolution, transposed convolution, fully connected layers
- Normalization layers: batch normalization layers (optimized through parameter fusion)
- Pooling layers: average pooling, max pooling
- Activation layers: ReLU, ReLU6, Sigmoid, etc. (implemented through polynomial approximation)
- Other operations: concatenation (Concat), summation (Add), nearest neighbor upsampling, etc.

#### Compiler Features

Our product compiler has significant advantages in computational correctness, execution efficiency, and usability, making it a key technical component for achieving automated encrypted inference.

**1. Computational Correctness Guarantee**

A core challenge in encrypted computation is noise management. CKKS ciphertexts naturally carry noise, and each homomorphic addition, multiplication, rotation, and other operations accumulate noise. When noise exceeds a preset threshold, decryption will produce incorrect results, leading to inference failure.

To manage noise levels, the compiler uses the ciphertext's level to measure its available multiplicative depth:

- Initial ciphertexts or refreshed ciphertexts have a level value greater than 0
- After ciphertexts pass through computational layers of the model, the level often decreases by 1 or more (depending on that layer's multiplicative depth consumption)
- When the level drops to 0, subsequent multiplication operations cannot be performed

The compiler ensures computational correctness throughout the entire inference process by inserting Bootstrapping computation nodes at appropriate positions in the computation graph to refresh the available multiplicative depth of ciphertexts.

Additionally, the compiler integrates other optimization techniques to ensure model inference accuracy, such as:

- Level alignment: Use drop level nodes to ensure input ciphertexts for summation or concatenation operations have the same level

**2. Execution Efficiency Optimization**

The computational overhead of encrypted inference is far higher than plaintext inference, making efficiency optimization crucial. The compiler needs to minimize overall computational and communication overhead while ensuring computational correctness.

Key factors affecting efficiency include:

- For the same homomorphic operation, the larger the input ciphertext's level, the greater the computational overhead (requiring processing of larger moduli)
- Bootstrapping operations have much higher computational overhead than ordinary homomorphic operations

Our product compiler uses intelligent search algorithms to automatically explore different ciphertext refreshing strategies and operator implementation schemes:

- Bootstrapping position optimization: While ensuring computational correctness, find the optimal combination of Bootstrapping insertion positions and frequency to minimize overall computation time
- Level consumption optimization: Prioritize executing homomorphic computation at low-level positions

Additionally, the compiler supports multiple multiplicative level consumption optimization techniques:

- Operator fusion: Fuse adjacent linear operations into a single computation to reduce additional multiplicative level consumption, such as fusion of batch normalization layers with convolutional layers
- Polynomial leading coefficient absorption: Convert polynomials to monic polynomials and absorb leading coefficients into adjacent linear layers to reduce multiplicative level consumption

**3. Automated Conversion Capability**

Traditional encrypted inference schemes often require developers to have deep cryptographic backgrounds, manually adjusting model structures, selecting homomorphic encryption parameters, designing data packing schemes, etc., with extremely high technical barriers and prone to errors.

Our product compiler achieves end-to-end automated conversion from plaintext models to secure inference models, greatly lowering the usage threshold:

- One-click conversion: Users only need to provide plaintext model files and target inference mode, and the compiler automatically completes all conversion work
- Automatic parameter configuration: The compiler automatically selects appropriate homomorphic encryption parameters based on model structure and security requirements
- Intelligent optimization: Automatically explores optimal computation graph structures without manual intervention

Specifically, the compiler's workflow includes the following key steps:

1. Model parsing: Input the original AI model (supporting PyTorch, ONNX, and other formats), extract the model's network structure and parameter information
2. Computation graph generation: Transform the model's forward inference process into a Directed Acyclic Graph (DAG) representation, with each operator corresponding to a computational node in the graph
3. Candidate scheme generation: Generate multiple candidate computation graphs with different structures based on the selected inference mode. For example, try different Bootstrapping insertion position combinations
4. Subgraph partitioning and cost evaluation: Partition each candidate computation graph into multiple subgraphs, evaluate overhead metrics such as computation time and communication volume for each subgraph
5. Optimal scheme selection: Synthesize cost evaluation results from each subgraph, calculate the overall overhead for each candidate scheme, and finally select the optimal scheme with minimal computational or communication overhead
6. Encrypted model output: Output the target encrypted model's computation graph structure, homomorphic encryption parameter configuration, and converted model parameter files for loading by the inference framework

Through this automated workflow, even AI engineers without cryptographic backgrounds can easily deploy their models as encrypted models supporting secure inference, greatly accelerating the application deployment of encrypted AI technology.

To more intuitively demonstrate the compiler's effectiveness, the following figure shows the secure inference model structure compiled from the MobileNetv2 plaintext model (FHE-BTP model). In the figure, "lv" is an abbreviation for level, and lv0 indicates that the ciphertext's level has dropped to 0, requiring bootstrapping operations to refresh the ciphertext level. To simplify the display, only computational nodes are shown in the figure, with data nodes omitted; the boxes of computational nodes are labeled with the current node's type and the level value of its output data node.

In the FHE-BTP mode secure inference model, all relu2d operators are replaced by second-order polynomial operators poly_relu2d, whose leading coefficients have been absorbed into adjacent layers, so each poly_relu2d operator only consumes one multiplicative depth.

<img src="../images/architecture/compiler_capacity.png" alt="compiler capacity" width="100%"/>

### Model Fine-tuning or Retraining

In encrypted inference scenarios, users typically want to directly use trained plaintext models for secure inference. However, as mentioned earlier, CKKS homomorphic encryption only natively supports addition and multiplication operations and cannot directly compute common nonlinear activation functions such as ReLU and SiLU. Therefore, it is necessary to replace activation functions with polynomial functions to support efficient Model Secure Inference.

However, simply replacing activation functions with polynomials often leads to significant model accuracy degradation. This is because polynomial approximations contain errors, and these errors continuously accumulate and amplify during propagation through multiple network layers. To solve this problem, the replaced model needs to be retrained or fine-tuned to make the network adapt to the new activation function form, thereby restoring the original inference accuracy.

Our product supports two activation function replacement and model retraining or fine-tuning schemes:

1. **AESPA^[Park, J., Kim, M. J., Jung, W., & Ahn, J. H. (2022). AESPA: Accuracy preserving low-degree polynomial activation for fast private inference. arXiv preprint arXiv:2201.06699. https://arxiv.org/abs/2201.06699] Method**: An existing method based on batch normalization and polynomial approximation fusion. After replacing activation functions, the replaced model needs to be retrained from scratch.
2. **Single-Stage Fine-tuning Method[^peregrine]**: An innovative method proposed by our product that can more quickly and efficiently convert pretrained deep neural networks into secure inference models. Compared to traditional methods, the single-stage fine-tuning method has significant advantages such as fewer training epochs, simple workflow, and minimal accuracy loss, while achieving low inference latency and low communication overhead.

Below we introduce our product's single-stage fine-tuning method. Its core is the proposed PolyAct-RN operator (Polynomial Activation with Range Normalization). This is a specially designed activation function replacement module used to replace traditional activation functions such as ReLU in encrypted inference.

The core idea of the PolyAct-RN operator is: during the training phase, adaptively constrain the dynamic input values of the activation function to a smaller bounded range (such as [-3, 3]), so that low-degree polynomials can achieve effective activation function approximation, and multiply back the adaptive scaling factor after polynomial approximation to restore output amplitude.

The ingenuity of this design lies in: it both ensures approximation accuracy (low-degree polynomials are sufficiently accurate within the small range after adaptive scaling) and controls computational overhead (only 2-4 degree polynomials are needed, consuming very little multiplicative depth).

During the inference phase, the PolyAct-RN operator fixes the scaling factor through running statistics accumulated during training, and behaves as a low-degree polynomial with fixed coefficients for the pre-activation values of each input channel. This means its coefficients do not depend on the specific values of input data and can be directly computed through homomorphic multiplication and addition in the encrypted state, being fully compatible with CKKS homomorphic encryption.

#### Single-Stage Fine-tuning Implementation Workflow

Our product's single-stage fine-tuning method requires only two simple steps to complete model conversion:

**Step 1: Model Preprocessing**

Make necessary structural adjustments to the original pretrained model according to the secure inference model structure provided by the compiler:

- Replace some or all nonlinear activation functions (such as ReLU, SiLU, etc.) with PolyAct-RN operators
- Replace some or all max pooling layers with average pooling layers
- Optionally adjust the model's input image size according to actual application requirements

**Step 2: Single-Stage Fine-tuning**

Fine-tune the preprocessed model using the target dataset, the network will automatically learn to adapt to the new activation function form, ultimately obtaining a secure inference model with restored accuracy.

The following figure illustrates obtaining a secure inference model using the single-stage fine-tuning scheme in Bootstrapping mode:

<img src="../images/algorithms/single-stage-fine-tuning-diagram.png" alt="single-stage fine-tuning diagram" width="40%"/>

#### Single-Stage Fine-tuning Characteristics

Compared to existing activation function replacement and model retraining methods, our product's single-stage fine-tuning method has the following significant advantages:

1. **Unity of High Accuracy and Low Latency**: Significantly reduces computational latency of encrypted inference while maintaining original model accuracy.

2. **Low Training Cost**: Only requires a small number of training epochs to complete model conversion and restore accuracy, greatly reducing computational resources and time costs

3. **Simple Workflow**: No need for complex multi-stage fine-tuning or sensitive hyperparameter tuning, simply replace original activation functions with PolyAct-RN operators and perform single-stage training

4. **Strong Architecture Generalization Capability**: The PolyAct-RN operator, as a plug-and-play activation function replacement module, can be seamlessly applied to various mainstream DNN architectures

Therefore, users can use the two activation function replacement schemes supported by our product to quickly convert existing plaintext models into high-accuracy, high-efficiency encrypted inference models, accelerating the practical deployment of encrypted AI technology in actual business applications.

## Deployment and Usage of Model Secure Inference Services

After completing model compilation and retraining, encrypted inference models can be deployed to actual business environments to provide users with privacy-protected AI inference services. This chapter introduces how to deploy the encrypted inference framework on the server side and how clients can use this service to complete secure inference. The entire workflow design follows the basic principle of "client-side encryption, server-side computation, client-side decryption" to ensure complete privacy protection of user data and inference results.

### Deployment Scheme

Our product's encrypted inference service adopts a client-server separation architecture, where the server is responsible for encrypted computation, and the client is responsible for key management and data encryption/decryption.

#### Server-Side Deployment

Model service providers need to deploy the following core components on the server:

1. Encrypted Inference Framework

This is the core runtime environment of the entire system, providing implementation of encrypted operators, encrypted computation scheduling, memory management, and other basic functions. The inference framework has been optimized to support multi-threaded CPU parallel computation and hardware acceleration (such as GPU accelerator).

2. Model Parameter Files and Model Computational Graph

Model weight parameters after retraining or fine-tuning, as well as network computational graph provided by compiler. These parameters will be encoded as CKKS plaintext polynomials during the deployment phase for use in encrypted computation.

Encrypted inference has significantly increased computational resource requirements compared to plaintext inference, especially memory consumption. Before deployment, it is necessary to evaluate the server's memory requirements based on the model's parameter count and selected homomorphic encryption parameters (such as polynomial modulus N). To improve online inference response speed, our product enables offline encoding preprocessing of model parameters during the deployment phase. This process pre-encodes all plaintext model parameters as CKKS plaintext polynomials and caches them.

#### Client-Side Deployment

The client side is relatively simple, requiring only deployment of lightweight SDK components that provide the following functions:

- Generate and manage CKKS key pairs
- Encode and encrypt input data
- Decrypt inference results
- Communicate with the server over the network

### Usage Workflow

After deployment is complete, encrypted inference services can begin. The entire inference workflow involves collaborative work between client and server, with the following diagram showing the complete interaction flow:

<img src="../images/architecture/eng-inference.png" alt="inference workflow" width="90%"/>

The entire workflow can be divided into client-side operations and server-side operations, explained in detail below.

#### Client (Requester) Workflow

The client is responsible for protecting data and obtaining inference results, ensuring that sensitive information remains in an encrypted state at all times.

**Step 1: Generate Key Pair**

The requester generates a public-private key pair and stores the private key for decrypting encrypted inference results.

**Step 2: Encrypt Input Data**

The requester uses the public key to encrypt data to be inferred. Taking an image classification task as an example, the client encodes image pixel values according to a specified packing scheme and encrypts them into a set of CKKS ciphertexts. The encryption process is completed locally on the client side, and original plaintext data never leaves the client device, thus ensuring data privacy.

**Step 3: Send Inference Request**

The client sends the encrypted ciphertext and public key together to the server and initiates an encrypted inference request.

**Step 4: Receive and Decrypt Results**

After the server completes encrypted inference computation, it returns encrypted inference results (also in CKKS ciphertext form). The client uses the previously saved private key to decrypt the result ciphertext and obtain the final inference results.

Throughout this process, the server can never know the client's original input data or final inference results, achieving end-to-end privacy protection.

#### Server (Service Provider) Workflow

The server is responsible for executing model inference computation in the encrypted state, without needing or being able to access the client's plaintext data.

**Step 1: Initialize Encrypted Inference Model**

Upon startup (or upon receiving the first inference request), the server needs to complete model initialization work:

- Load the secure inference model's computational graph and parameter files
- Perform CKKS plaintext encoding of model weights according to homomorphic encryption parameter configuration generated by the compiler
- Initialize various encrypted operators (convolution, activation, pooling, etc.)

If offline preprocessing is enabled, the next step will directly load pre-encoded parameters; otherwise, real-time encoding is required.

**Step 2: Execute Encrypted Inference Computation**

After receiving the public key and encrypted ciphertext sent by the client, the server completes inference according to the following workflow:

- Execute encrypted operator computations (convolution, activation, pooling, etc.) layer by layer according to the computation graph structure provided by the compiler
- Execute Bootstrapping operations at necessary positions according to the computation graph structure provided by the compiler to refresh the available multiplicative depth of ciphertexts
- Complete computation of the entire computation graph and return inference result ciphertext

After computation is complete, the server returns the encrypted inference results to the client. Throughout the entire computation process, the server can only see ciphertext data and cannot know any plaintext information.

In the above workflow, our product's secure inference service provides the following security guarantees:

- Client data privacy: Input data is encrypted on the client side before being sent; the server cannot decrypt it
- Inference result privacy: Inference results are returned in ciphertext form; only the client holding the private key can decrypt them
- Model parameter protection: Model parameters are stored on the server; the client cannot obtain model structure and parameters
- Computation process privacy: All intermediate computation results are ciphertext; the server cannot infer input or output information from them

Through this design, our product protects both client data privacy and the service provider's model intellectual property, achieving bidirectional privacy protection.

## Performance Evaluation

We conducted comprehensive performance testing on multiple standard benchmark datasets in the computer vision domain, evaluating the framework's practicality from multiple dimensions including model accuracy and inference latency.

This chapter will present detailed performance results in typical application scenarios such as image classification and object detection, and through comparison with plaintext models, demonstrate that our framework can achieve high-accuracy and high-efficiency model inference while ensuring privacy security.

### Testing Environment

To ensure objectivity of test results, we conducted all experiments under unified hardware configurations:

Hardware Configuration:

- **Server**:
  - CPU: Intel(R) Xeon(R) Gold 6226R @ 2.90GHz, 32 cores, 256GB memory
  - GPU: NVIDIA RTX 5880 Ada Generation, 48GB memory
- **Client**:
  - CPU: Intel(R) Xeon(R) Gold 6226R @ 2.90GHz, 32 cores, 256GB memory

For some testing tasks, we also provide GPU hardware acceleration performance data to demonstrate the acceleration effect of dedicated hardware on encrypted inference.

### Performance Metrics Explanation

Performance evaluation of encrypted inference cannot focus solely on inference speed but must comprehensively consider multiple dimensions including model accuracy and computational overhead. We define the following key performance metrics:

- **Baseline Model Accuracy**: The inference accuracy of the original plaintext model on the corresponding dataset, serving as a comparison baseline
- **Secure Inference Model Accuracy**: The inference accuracy of the secure inference model on the same dataset after activation function replacement and retraining. Ideally, this accuracy should be comparable to the baseline model accuracy, indicating that privacy protection measures have not significantly compromised the model's practicality
- **Inference Latency**: The total time consumed by the server in executing Fully Homomorphic Encryption (FHE) computation, including homomorphic operations on ciphertexts, Bootstrapping operations, etc. This metric reflects the computational overhead of encrypted computation

Metric Interpretation:

- Inference Latency determines end-to-end inference latency
- The closeness between secure inference model accuracy and baseline accuracy reflects the impact of privacy protection technology on model practicality

To comprehensively verify the effectiveness and practicality of this framework, we conducted thorough experimental evaluations on multiple standard benchmark datasets in the computer vision domain, covering typical application scenarios such as image classification and object detection. Experimental results demonstrate that this framework can achieve inference accuracy comparable to plaintext models while ensuring privacy security, and significantly improve inference efficiency through algorithm optimization and hardware acceleration, making encrypted inference feasible in actual business applications.

### Image Classification Tasks

#### CIFAR-10 Dataset

For image classification tasks, we first conducted tests on the CIFAR-10 dataset. This dataset is one of the most classic image classification benchmarks in the deep learning field, containing 60,000 32×32 pixel color images uniformly distributed across 10 categories, with 50,000 used for training and 10,000 for validation.

We tested using the ResNet20 model. Our framework's encrypted inference model input size remains consistent with the original images (32×32), requiring no image scaling preprocessing. We tested the encrypted inference performance of this model under FHE+Bootstrapping mode (N=65536). Performance data is shown in the table below:

**FHE+Bootstrapping Mode (N=65536)**

| Model | Parameters (M)| Baseline Model Accuracy| Secure Inference Model Accuracy | Computation Resource | Inference Latency (s) |
| :---: | :---: |  :---: |  :---: | :---: | :---: |
| ResNet20 | 0.3 | 91.9% | 92.0% | 16-thread CPU | 310.3 |
| ResNet20 | 0.3 | 91.9% | 92.0% | GPU | 13.5 |

#### ImageNet Dataset

ImageNet is the standard benchmark dataset for large-scale visual recognition, with the ILSVRC 2012 classification task containing 1,000 categories, a training set of approximately 1.28 million images, and a validation set of 50,000 images. Our framework's encrypted inference model uses the MobileNetv2 architecture with input image size of 256×256.

ImageNet is the most influential large-scale visual recognition benchmark dataset in the computer vision field. The ILSVRC 2012 classification task is its core competition project, containing 1,000 fine-grained categories, a training set of approximately 1.28 million images, and a validation set of 50,000 images. The scale and complexity of this dataset make it the gold standard for evaluating the generalization capability of deep learning models, and it is also a key test for verifying the practicality of encrypted inference frameworks in large-scale, high-resolution scenarios.

This framework tested using the MobileNetv2 architecture, with this model having 3.5M parameters and achieving 71.8% plaintext baseline accuracy on ImageNet, representing mainstream performance levels for lightweight networks. After training with our product's single-stage fine-tuning method, the secure inference model achieves 70.1% accuracy, comparable to the baseline model accuracy, fully validating the effectiveness of our activation function replacement strategy on large-scale datasets. The input image size used in testing is 256×256 pixels. We tested the encrypted inference performance of this model under FHE+Bootstrapping mode (N=65536). Performance data is shown in the table below:

**Bootstrapping Mode (N=65536)**

| Model | Parameters (M)| Baseline Model Accuracy| Secure Inference Model Accuracy | Computation Resource | Inference Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetv2 | 3.5 | 71.8% | 70.1% | 16-thread CPU | 1090.2 |
| MobileNetv2 | 3.5 | 71.8% | 70.1% | GPU | 71.3 |

### Object Detection Tasks

#### Blood Cell Detection Dataset

Object detection is another important application domain in computer vision. Compared to image classification tasks, object detection not only needs to identify object categories in images but also precisely localize object positions. In sensitive domains such as medical image analysis, patients' blood sample images contain substantial privacy information, and using encrypted inference technology can complete automated detection and diagnosis while protecting patient privacy.

We validated this framework's performance on object detection tasks using the Blood Cell Count Detection (BCCD) dataset. This dataset is used for automatic detection, classification, and counting of red blood cells, white blood cells, and platelets in Complete Blood Count (CBC), a typical application scenario for medical image AI-assisted diagnosis. To improve the model's generalization capability, this dataset has been augmented with data enhancement techniques including rotation, flipping, brightness adjustment, and other transformations, enabling the model to adapt to different shooting conditions and sample states.

This framework tested using the YOLOv5n architecture, which is the lightweight version of the YOLOv5 series with 1.9M parameters. The plaintext baseline model achieves 92.2% mAP@0.5 on this dataset. After our product's activation function replacement and retraining, the secure inference model achieves 92.0% mAP@0.5, with accuracy loss of only 0.2 percentage points, fully demonstrating this framework's effectiveness on object detection tasks. The input image size used in testing is 512×512 pixels. We tested the encrypted inference performance of this model using 16-thread CPU under FHE+Bootstrapping mode (N=65536), with performance data shown in the table below.

**About the mAP@0.5 Evaluation Metric**

Since the evaluation method for object detection tasks differs from classification tasks, we briefly explain the meaning of the mAP@0.5 metric here. mAP@0.5 (mean Average Precision at IoU threshold 0.5) is the standard evaluation metric in the object detection field, used to comprehensively measure model detection accuracy and localization precision:

- IoU (Intersection over Union): The ratio of the intersection area to the union area between the predicted bounding box and the ground truth bounding box, used to measure localization accuracy
- @0.5: Indicates the IoU threshold is set to 0.5, meaning a prediction is considered correct only when the IoU between the predicted box and ground truth box is ≥ 0.5 and the predicted category matches the ground truth category
- mAP (mean Average Precision): Calculate Average Precision (AP) for each category separately, then average across all categories to obtain the model's average detection accuracy on the entire dataset

**FHE+Bootstrapping Mode (N=65536)**

| Model | Parameters (M) | Baseline Model mAP@0.5 | Secure Inference Model mAP@0.5 | Computation Resource | Inference Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5n | 1.9 | 92.2% | 92.0% | 16-thread CPU | 1519.3 |

[^peregrine]: Ling, H., Wang, Y., Chen, S., & Fan, J. (2025). Peregrine: One-Shot Fine-Tuning for FHE Inference of General Deep CNNs. arXiv preprint arXiv:2511.18976. https://arxiv.org/abs/2511.18976
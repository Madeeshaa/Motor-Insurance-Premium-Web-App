# Methodology

This chapter explains the methodology used in this study to achieve the objectives outlined in the introduction. The main aim of the research is to develop a neuro fuzzy framework for estimating vehicle insurance premiums by modeling risk and expected loss under uncertainty. To achieve this objective, a structured modeling approach was followed that combines traditional actuarial methods with modern computational techniques.

## 1. Data Preprocessing and Feature Engineering
Before training any models, the raw dataset must be properly cleaned and transformed to ensure that the mathematical formulas work effectively.

**Variable Capping and Creation:**
Important driver and vehicle characteristics definitions were adjusted to fit realistic ranges. For example, driver age was restricted to be between 18 and 85 years, and driving experience was limited to a maximum of 60 years. 

To improve the models' learning capabilities, new features were created to represent interactions between different risk factors mathematically. The key mathematical equations used to generate these interactions are:

- **Logarithm of Vehicle Value:** Huge monetary values can cause instability in mathematical models, so the standard mathematical natural log function is applied to smooth out huge numbers.
$$ \text{log\_Value\_vehicle} = \ln(1 + \text{Value\_vehicle}) $$

- **Age to Experience Ratio:** A younger driver with less experience is statistically vastly different from an older driver with the exact same experience.
$$ \text{age\_exp\_ratio} = \frac{\text{Policyholder\_age}}{\text{Driving\_Experience} + 1} $$

- **Engine Power against Vehicle Age:** 
$$ \text{cc\_per\_veh\_age} = \frac{\text{Cylinder\_capacity}}{\text{Vehicle\_age} + 1} $$

- **Multiplicative Interactions:**
Further features simply multiplied characteristics to analyze joint risks.
$$ \text{age\_x\_veh\_age} = \text{Policyholder\_age} \times \text{Vehicle\_age} $$
$$ \text{exp\_x\_log\_val} = \text{Driving\_Experience} \times \text{log\_Value\_vehicle} $$

**Handling Extreme Claim Costs:**
In insurance, extremely rare but massive accidents (such as multi-million dollar collisions) can completely confuse a predictive model. To solve this, claim costs were mathematically "winsorized". This means that the absolutely highest 0.5% of extreme claim costs were simply reduced and capped at the 99.5th percentile mark to prevent model collapse without completely ignoring large accidents.

## 2. Classical Actuarial Baseline Construction (GLM)
In standard practice, the basic cost of insurance (pure premium) for a single policyholder is estimated mathematically by predicting the predicted number of claims (frequency) separated from the expected cost of those individual claims (severity), and then multiplying them. A Generalized Linear Model (GLM) was built to act as the conservative baseline to test our newer models against.

**Claim Frequency:** 
Predicting the number of accidents a driver will likely have is modeled precisely using a **Negative Binomial distribution**. This is very useful because insurance claim counts are usually "overdispersed", meaning there is higher variance than the average number mathematically expects.
$$ \log(\lambda_i) = X_i \beta $$
Where $\lambda_i$ is the expected frequency for a policyholder $i$, $X$ represents the input facts (age, value), and $\beta$ represents the learned risk coefficients.

**Claim Severity:**
The actual expected monetary cost of a granted claim is predicted deeply using a **Gamma distribution** combined with a mathematical log function. This is necessary because claim costs cannot drop below zero and tend to heavily skew towards massive right-sided tails on a graph.
$$ \log(\mu_i) = X_i \gamma $$

The final baseline cost prediction ($\hat{P}_i$) for each driver is then multiplied securely together:
$$ \hat{P}_i = \lambda_i \times \mu_i $$

## 3. Neuro-Fuzzy Premium Model (ANFIS)
Traditional linear models are completely built upon strict logical boundaries, which is heavily flawed when talking about fluid human definitions like a "young driver" or an "old vehicle." To successfully escape this, an Adaptive Neuro-Fuzzy Inference System (ANFIS) was newly deployed. This unique framework actively combines fuzzy logic (representing flexible human reasoning) with deep neural learning capabilities.

### 3.1 Fuzzification and Membership Function Development
"Fuzzification" takes exact basic numbers—like a policyholder being explicitly 24 years old—and assigns them partial percentage degrees across multiple flexible overlapping risk categories (e.g., scoring 60% Young, and 40% Adult at the exact same time). Five core variables were fuzzified: Age, Experience, Vehicle Age, Vehicle Value, and Engine Capacity. 

To accomplish this mathematical translation practically, the active model employs sliding **Gaussian Membership Functions**, mathematically written within the scripts as:
$$ \mu(x) = \exp\left(-\frac{1}{2} \left(\frac{x - mean}{\sigma + 10^{-4}}\right)^2\right) $$
Where $mean$ is the calculated center of the defined category (like the peak age of an "Adult"), $\sigma$ defines naturally how extremely wide the category spreads outward, and $10^{-4}$ is a very tiny safety number (epsilon) added underneath solely to prevent the backend computers from accidentally dividing a zero and crashing down during deep calculations.

### 3.2 Artificial Neural Network Optimization

The membership values obtained from the earlier fuzzification process were used directly as inputs to a PyTorch neural network component. Once translated into these fuzzy group combinations (totaling 243 overlapping theoretical rules mathematically), the neural network layers actively learn the deep relationships between the fuzzy risk representations and the overall risk level through data-driven training.

Within this structural framework, the network contains dual parallel sections specifically designed to simultaneously estimate two outputs: the expected claim frequency and the expected claim severity. These unique predictions are obtained through nonlinear transformations within the neural network's layers, enabling the model to accurately capture complex interactions among the various risk variables. 

To achieve this, the model parameters, crucially including both the neural network weights and the earlier Gaussian membership function parameters, were comprehensively optimized together using gradient-based learning techniques structurally built into PyTorch. This continuous joint optimization allows the system to automatically adjust both the foundational fuzzy representations and the predictive model layers simultaneously in order to minimize prediction error.

Furthermore, the trained network was specifically optimized not just to minimize standard error, but to properly rank driver risks progressively using a novel **Pairwise Ranking Loss** mathematical equation. This specialized code looks at thousands of random pairs of drivers rapidly and actively penalizes the overall network heavily if it mistakenly ranks a safe driver's expected premium vastly higher than a genuinely dangerous driver's basic premium. Finally, the final predicted pure premium for each policy was naturally obtained by safely combining the predicted claim frequency and predicted claim severity.

## 4. Stacked Ensemble Model

No single predictive algorithm flawlessly understands all data patterns perfectly. Traditional GLMs excel at identifying broad, global mathematical trends, while the Neuro-Fuzzy framework is specifically built to capture hyper-local, complex human risk boundaries. To extract the best mathematically possible prediction safely from both, an advanced structural process called "Stacking" was executed. 

Directly blending models that have already seen the entire training data typically leads to severe mathematical "overfitting", where the system merely memorizes past accidents rather than predicting future risks. To completely avoid this leakage, an unbiased validation technique called **5-Fold Cross-Validation** was employed. The training data was split into five equal parts. The GLM and ANFIS base models were repeatedly trained on four parts to uniquely generate pure, unseen predictions recursively on the fifth part. This securely generates 'Out-of-Fold' (OOF) base predictions for the entire dataset deliberately without ever letting the algorithms look at the specific testing facts beforehand. 

Instead of using a static average or a simple linear formula to combine the base models, a higher-level supervisor artificial intelligence layer programmed as a deep **Multi-Layer Perceptron (MLP) Neural Stacker** was introduced. This top-tier meta-learner receives a rich dataset of inputs to make its final judgment:
- The pure OOF premium predictions directly from both the GLM and ANFIS models.
- Their logarithmic mathematical equivalents, shrinking massive numeric gaps to help the neural network learn smoothly.
- The absolute calculated mathematical differences between the GLM and ANFIS outputs, actively forcing the MLP to study precisely where and by how much the two base models strongly disagree.
- The original initial input features (Age, Vehicle Value, etc.), allowing the meta-learner to recognize complex contexts. 

Armed with this data, the MLP top neural network automatically studies exactly which conditions cause the GLM to structurally fail locally and precisely where the Neuro-Fuzzy logic uniquely shines. It ultimately determines an optimal non-linear blending combination dynamically, drastically increasing the predictive power over a basic static mixture.

In the practical business of insurance, merely ranking drivers correctly is mathematically insufficient to sustain a company; the total money collected must accurately cover the total money lost. Because neural networks focus on individual accuracy, they can occasionally drift away from the actual collective sum of real-world financial costs. After the stacker makes a logical finalized decision for all individuals, the entire column of combined premiums is passed actively through a mathematical **Isotonic Regression**. This acts purely as a final numerical calibration safety net. It strictly forces the combined collective grand sum of all the new predicted premiums to perfectly and precisely equal the genuinely true historical total mass of past financial losses physically experienced earlier by the insurer. This guarantees absolute financial solvency while perfectly maintaining the complex, highly optimized risk-ranking order discovered by the MLP stacker.

## 5. Survival Simulation
Financial driver risk stretches across the continuous invisible passage of months. To easily visualize this progressive degradation of driving safety smoothly, practical mathematical survival simulations were algorithmically rendered for specific risk profiles.

The survival function reliably models the exact continuous probability ($S(t)$) that a selected policyholder systematically completes a given amount of time $t$ smoothly without submitting a disastrous claim. For this specific study, the **Exponential Survival Function** was exclusively chosen:
$$ S(t) = e^{-\lambda t} $$

Where $e$ is a standard mathematical constant, $t$ is the exact time passed (e.g., years), and $\lambda$ is the unique expected claim frequency specifically calculated by the AI models beforehand for that exact driver. This specific function guarantees perfect mathematical harmony with the rest of the framework. Because the deep neural networks and baseline GLMs were built specifically to predict claim frequency totals using Poisson mathematics, pure probability theory dictates that the continuous timeline between those exact events inherently follows an Exponential curve. Furthermore, this curve accurately reflects human driving by assuming a driver's fundamental base skill level remains broadly steady throughout a standard one-year contract, confidently avoiding the unnecessary timeline complexities found in other models strictly designed for physical aging or mechanical decay. Relying purely on the single calculated risk parameter ($\lambda$) also safely prevents the computer from over-complicating simulations or guessing mathematically unnecessary crash acceleration trends.

## 6. Model Validation and Comparison
A distinctly separate reserved pure testing data set mathematically comprising exactly an untouched 20% block of the whole original factual information was completely isolated and entirely hidden fully during the model-building stages safely. Standard basic mathematics then strictly provides strong specific metrics to evaluate models confidently.

- **Mean Absolute Error (MAE):** The average regular dollar difference found strictly between the computer quote basically ($\hat{y}_i$) and the actual human payout truth ($y_i$) precisely. It provides a straightforward linear mathematical penalty for calculation errors across $N$ total test policies.
  $$ MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$

- **Root Mean Squared Error (RMSE):** The average squared financial difference uniquely calculated, intensely penalizing models for getting massive disastrous claims severely wrong fundamentally. By mathematically squaring the exact differences before comprehensively averaging them, the equation aggressively forces the framework to prioritize avoiding uniquely gigantic mistakes.
  $$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$

- **Gini Index:** The basic gold standard evaluation metric used securely across the insurance business. It solidly determines the complex algorithm's overall power systematically to successfully cleanly rank simple drivers strictly by their genuine realized physical risk perfectly, cleanly independent totally of simple raw physical monetary amounts logically. Mathematically, it fundamentally evaluates the exact area securely situated beneath the model's dynamically plotted Lorenz curve ($L(p)$), where $p$ represents the true cumulative proportion of directly ranked policies.
  $$ Gini = 1 - 2 \int_{0}^{1} L(p) \, dp $$
  A final mathematical value of 0 rigorously dictates complete numerical randomness mathematically, whereas a higher percentage forcefully confirms a powerful absolute ranking capability cleanly pulling historically perfectly safe drivers safely away from mathematically dangerous ones.

**Visual Evaluation:** 
Lorenz curves dynamically plot mathematically the cumulative sequential proportion of policies directly strictly against the actual cumulative proportion of historical observed physical currency cleanly lost truthfully. Decile lift tables mathematically efficiently physically split the total ranked testing dataset exactly identically into ten simple ordered ranking buckets neatly, ultimately fundamentally verifying distinctly whether the aggressively blended AI Stacker methodology reliably separates risky dangerous drivers fully away from the safe clean population clearly.

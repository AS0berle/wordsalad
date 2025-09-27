## What are Vector Symbolic Architectures?

Vector Symbolic Architectures (VSA) are a class of approaches which treat vectors as symbols and use mathematical operators to perform symbolic manipulation on them. Each VSA defines the following:

- A vector sampling distribution $\Omega$
- A binding operator $B(\cdot, \cdot)$
- A bundling operator $+$ (usually addition)
- A retrieval (or unbinding) operator $B^*(\cdot, \cdot)$  

To encode information, symbols are bound in key/value (or role/filler) pairs, and then bound pairs are bundled together into a hypervector. Then, using any original key, we can use the retrieval operator to get its corresponding value vector back out of the hypervector.

As an example, consider the sentence *“Alex likes chocolate”*. We can assign vectors from the VSA’s distribution to the roles $\\{First, Second, Third\\}$ and values $\\{Alex, likes, chocolate\\}$.

We can then represent the entire sentence with:

$$
S = B(First, Alex) + B(Second, likes) + B(Third, chocolate)
$$

Then, we can later recover the $First$ word from the sentence with:

$$
B^*(S, First) = \widehat{Alex} \approx Alex
$$

In general, this retrieval process is subject to noise; a “good” VSA is one that is less noisy, in the sense that $\widehat{Alex}$ is very close to $Alex$.

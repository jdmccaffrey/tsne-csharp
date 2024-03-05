# tsne-csharp
Implementation of t-SNE visualization using C#.

The basic form of the t-SNE ("t-distributed Stochastic Neighbor Embedding") technique is very specific. It starts with source data that has n rows and 3 or more colmns, and creates a reduced dataset with n rows and 2 columns. The reduced data can be used to create an XY graph where the first column is used as the x values and the second column is used as the y values.

This C# implementation is a (mostly) direct translation of the original Python implementation written by researcher L. van der Maaten, who co-authored the original t-SNE paper. The original paper is "Visualizing Data using t-SNE" (2008), L. van der Maaten and G. Hinton. A link to the original Python implementation of t-SNE is at https://lvdmaaten.github.io/tsne/.

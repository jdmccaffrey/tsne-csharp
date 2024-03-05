# tsne-csharp
Implementation of t-SNE visualization using C#.

The basic form of the t-SNE ("t-distributed Stochastic Neighbor Embedding") technique is very specific. It starts with source data that has n rows and 3 or more colmns, and creates a reduced dataset with n rows and 2 columns. The reduced data can be used to create an XY graph where the first column is used as the x values and the second column is used as the y values.

This C# implementation is a (mostly) direct translation of the original Python implementation written by researcher L. van der Maaten, who co-authored the original t-SNE paper. The original paper is "Visualizing Data using t-SNE" (2008), L. van der Maaten and G. Hinton. A link to the original Python implementation of t-SNE is at https://lvdmaaten.github.io/tsne/.

## Demo Data

The t-SNE implementation was tested using a tiny 12-item subset of the Penguin Dataset. The data is:

0, 39.5, 17.4, 186, 3800  
0, 40.3, 18.0, 195, 3250  
0, 36.7, 19.3, 193, 3450  
0, 38.9, 17.8, 181, 3625  
1, 46.5, 17.9, 192, 3500  
1, 45.4, 18.7, 188, 3525  
1, 45.2, 17.8, 198, 3950  
1, 46.1, 18.2, 178, 3250  
2, 46.1, 13.2, 211, 4500  
2, 48.7, 14.1, 210, 4450  
2, 46.5, 13.5, 210, 4550  
2, 45.4, 14.6, 211, 4800  

Each line represents a penguin. The fields are species, bill length, bill width, flipper length, body mass. The species labels in the first column are not directly used by the t-SNE reduction function -- they are used only in a graph to verify that the reduced data accuractely reflects the source data.

## Usage

Calling code looks like:

    string ifn = @"C:\VSM\TSNE\Data\penguin_12.txt";  
    double[][] X = TSNE.MatLoad(ifn, new int[] { 1, 2, 3, 4 }, ',', "#"); // not col [0]  
    Console.WriteLine("Source data: ");  
    TSNE.MatShow(X, 1, 10, true); // 1 decimal, show indices  

    int maxIter = 500;  
    int perplexity = 3;  
    double[][] reduced = TSNE.Reduce(X, maxIter, perplexity);  

    Console.WriteLine("Reduced data: ");  
    TSNE.MatShow(reduced, 2, 10, true);  

    Console.WriteLine("Saving reduced data for a graph ");  
    string ofn = @"C:\VSM\TSNE\Data\penguin_reduced.txt";  
    TSNE.MatSave(reduced, ofn, ',', 2); // comma delim,  2 decimals  

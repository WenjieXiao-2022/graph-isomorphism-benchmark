/// Converts SivaLab graph format into Matlab adjacency matrix.
///   Several strongly regular graphs can be found here: https://sites.google.com/site/giconauto/home/benchmarks

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

typedef unsigned short int uint16_t; 

bool isInvalid(char c)
{
    return c == ')';
}

// read one uint16_t
uint16_t readWord(ifstream& in)
{
    unsigned char bytes[2];
    in.read((char*)bytes, 2);          // read 2 bytes
    return bytes[0] | (bytes[1] << 8); // construct the 16-bit value
}

// read graph from SivaLab input file
void readGraph(string fileName)
{
    ifstream is(fileName.c_str(), ios_base::in | ios_base::binary);
    if (!is.is_open())
    {
        cerr << "Cannot open input file " << fileName << endl;
        return;
    }
    
    uint16_t nVertices = readWord(is);
    cout << "n = " << nVertices << "\n";
    
    // generate adjacency matrix
    vector<vector<int> > A(nVertices);
    for (int i = 0; i < nVertices; ++i)
        A[i].resize(nVertices, 0);

    uint16_t nEdges, v;
    for (int i = 0; i < nVertices; ++i)
    {
        nEdges = readWord(is);
        // cout << i << ": " << nEdges << " edges\n";
        for (int j = 0; j < nEdges; ++j)
        {
            v = readWord(is);
            A[i][v] = 1;
            // cout << "(" << i << ", " << v << ")\n";
        }
    }
    is.close();

    // print adjacency matrix
    fileName += ".m";
    replace(fileName.begin(), fileName.end(), '-', '_');
    replace(fileName.begin(), fileName.end(), '(', '_');
    fileName.erase(std::remove_if(fileName.begin(), fileName.end(), &isInvalid), fileName.end());

    ofstream os(fileName.c_str());
    os << "n = " << nVertices << ";\nA = [\n";
    for (int i = 0; i < nVertices; ++i)
    {
        for (int j = 0; j < nVertices; ++j)
        {
            os  << " " << A[i][j];
            if (j == nVertices-1)
                os  << "\n";
        }
    }
    os  << "];\n";
}

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        cout << argv[i] << "\n";
        readGraph(argv[i]);
    }
}

#include <iostream>
#include <cmath>
#include <vector>
#include <string.h>

using namespace std;

// amount of dimensional
const int d = 2;
const int N = 100;

/// based on https://www.mvstat.net/mvksa/mvksa.pdf

/// Matrix always is
/// (A, B)
/// (C, D)

/***
 * |H| - determinant of bandwidth matrix
 * H ^ (1/2) - squared matrix
 ***/

void print(float x) {
    cout << x << endl;
}

void print(vector<float> v) {
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "\n";
}

void printMatrix(float** matrix, const int lines, const int columns) {
    cout << "-----------\n";
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < columns; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "-----------\n";
}

void print2dMatrix(float **matrix) {
    printMatrix(matrix, 2, 2);
}

// return matrix lines x columns
float** getMatrix(const int lines, const int columns) {
    float** matrix = new float * [lines];
    for (int i = 0; i < lines; i++) {
        matrix[i] = new float [columns];
    }
    return matrix;
}

float max(vector<float> x) {
    int len = x.size();
    if(len == 0) {
        cout << "Error in max function vector length = 0" << endl;
        exit(-1);
    }
    float maximum = x[0];
    for (int i = 1; i < len; i++) {
        if(x[i] > maximum) {
            maximum = x[i];
        }
    }

    return maximum;
}

float min(vector<float> x) {
    int len = x.size();
    if(len == 0) {
        cout << "Error in max function vector length = 0" << endl;
        exit(-1);
    }
    float minimum = x[0];
    for (int i = 1; i < len; i++) {
        if(x[i] < minimum) {
            minimum = x[i];
        }
    }

    return minimum;
}

// return 2x2 matrix
float** get2dArray() {
    return getMatrix(2, 2);
}

// return |H| - determinant of 2d matrix
float determinant2d(float matrix[2][2]) {
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

// return |H| - determinant of 2d matrix
float determinant2d(float** matrix) {
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

/***
 * find H ^ (1/2)
 * https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix#:~:text=A%20square%20root%20of%20a,obtained%20by%20an%20explicit%20formula.
***/
float **getSquareRoot2dMatrix(float** matrix) {
    float** squareRoot2dMatrix = get2dArray();

    float determinantOfH = determinant2d(matrix);

    if (determinantOfH < 0) {
        cout << "Error: determinant of matrix < 0" << endl;
        exit(-1);
    }

    float s1 = sqrt(determinantOfH);

    // A + D
    float r = matrix[0][0] + matrix[1][1];
    float sum = r + 2 * s1;

    if (sum < 0) {
        cout << "Error: sum" << endl;
        exit(-1);
    }

    float t = sqrt(sum);

    if (t == 0) {
        cout << "Error: t == 0" << endl;
        exit(-1);
    }

    // (A + s) * (1/t)
    squareRoot2dMatrix[0][0] = (matrix[0][0] + s1) / t;
    // B * (1/t)
    squareRoot2dMatrix[0][1] = matrix[0][1] / t;
    // C * (1/t)
    squareRoot2dMatrix[1][0] = matrix[1][0] / t;
    // (D + s) * (1/t)
    squareRoot2dMatrix[1][1] = (matrix[1][1] + s1) / t;

    return squareRoot2dMatrix;
}

/***
 * https://www.mathsisfun.com/algebra/matrix-inverse.html
 * @param matrix
 * @return matrix^-1
 ***/
float** getInverse2dMatrix(float **matrix) {
    float** inverse2dMatrix = get2dArray();

    float determinant = determinant2d(matrix);

    if (determinant == 0) {
        cout << "Error: determinant of matrix == 0" << endl;
        cout << "This Matrix has no Inverse" << endl;
        exit(-1);
    }

    // D * (1 / determinant)
    inverse2dMatrix[0][0] = matrix[1][1] / determinant;
    // -B * (1 / determinant)
    inverse2dMatrix[0][1] = -matrix[0][1] / determinant;
    // -C * (1 / determinant)
    inverse2dMatrix[1][0] = -matrix[1][0] / determinant;
    // A * (1 / determinant)
    inverse2dMatrix[1][1] = matrix[0][0] / determinant;

    return inverse2dMatrix;

}

float** toMatrixFromVector(float* vector, const int len) {
    float** matrix = new float* [1];

    for (int i = 0; i < len; i++) {
        matrix[i] = new float [len];
    }

    for (int i = 0; i < len; i++) {
        matrix[0][i] = vector[i];
    }

    return matrix;
}

// .T
float** t(float* vector, const int len) {
    float** tVector = new float* [len];

    for (int i = 0; i < len; i++) {
        tVector[i] = new float [1];
    }

    for (int i = 0; i < len; i++) {
        tVector[i][0] = vector[i];
    }

    return tVector;
}

float** toMatrix(vector<float> x) {
    int len = x.size();
    float** tVector = new float* [len];

    for (int i = 0; i < len; i++) {
        tVector[i] = new float [1];
    }

    for (int i = 0; i < len; i++) {
        tVector[i][0] = x[i];
    }

    return tVector;
}

void printTvector(float** vector, int len) {
    for (int i = 0; i < len; i++) {
        cout << vector[i][0] << endl;
    }
}

float** multiplyMatrix(float** matrix1, float** matrix2,
        const int lines1, const int columns1, const int lines2, const int columns2) {

    float** matrix3 = getMatrix(lines1, columns2);

    for(int i = 0; i < lines1; i++){
        for(int j = 0; j < columns2; j++){
            for(int a = 0; a < columns1; a++){
                matrix3[i][j] += matrix1[i][a] * matrix2[a][j];
            }
        }
    }

    return matrix3;

}

void multiplyMatrixToNumber(float** matrix, const int l, const int c, float number) {
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < c; j++) {
            matrix[i][j] /= number;
        }
    }
}

// x - vector
// work only for 2d
// TODO pow don't work with fractional numbers!
// for dimensional = 2 is OK
float normalKernelFunction2d(float* x, const int lenX) {

    float** resultMatrix = multiplyMatrix(t(x, lenX), toMatrixFromVector(x, lenX), 1, lenX, lenX, 1);
    multiplyMatrixToNumber(resultMatrix, lenX, lenX, 2);

//    float** Kx = pow(2 * M_PI, d / 2) * exp(resultMatrix);

    // TODO exp of matrix
    // mb use it https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_exp

}

float** sumOfMatrix(float** matrix1, float** matrix2, const int l, const int c) {
    float** resultMatrix = getMatrix(l, c);

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < c; j++) {
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return resultMatrix;
}

float factorial(int number) {

    if (number == 0 || number == 1) {
        return 1;
    }

    int result = 1;

    for (int i = 1; i <= number; i++) {
        result *= i;
    }

    return result;

}

// return Identity matrix
float** getEmatrix(const int dim) {
    float** Ematrix = getMatrix(dim, dim);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                Ematrix[i][j] = 1;
            }
        }
    }

    return Ematrix;

}

// it's more quickly I'm think
float** get2dEmatrix() {
    float** Ematrix = get2dArray();
    Ematrix[0][0] = 1;
    Ematrix[1][1] = 1;
    return Ematrix;
}

// https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Normal.html
float dnorm(float x) {
    // dnorm(0) == 1/sqrt(2*pi)
    if (x == 0) {
        // TODO mb just return 0.398942
        return 1 / sqrt(2 * M_PI);
    }

    // TODO for x != 0 but it no needed now
}

// https://rdrr.io/cran/ks/src/R/normal.R
// TODO need to check
float dnormDeriv(float x=0, float mu=0, float sigma=1, float deriv_order=0) {
    float r = deriv_order;
    float phi = dnorm(0);

    float X = x - mu;
    float arg = x / sigma;

    float hmold0 = 1;
    float hmold1 = arg;
    float hmnew = 1;

    if (r == 1) {
        hmnew = hmold1;
    }
    else if (r >= 2) {
        // (2:r) == r - 1
        for (int i = 0; i < r - 1; i++) {
            hmnew = arg * hmold1 - (i - 1) * hmold0;
            hmold0 = hmold1;
            hmold1 = hmnew;
        }
    }

    float derivt = pow(-1, r) * phi * hmnew / pow(sigma, r);

    return derivt;
}

float mean(vector<float> x) {
    float sum = 0;
    for (int i = 0; i < x.size(); i++) {
        sum += x[i];
    }
    return sum / x.size();
}

// sample variance
float S2(vector<float> x) {
    float m = mean(x);
    float sum = 0;

    for (int i = 0; i < x.size(); i++) {
        float diff = x[i] - m;
        sum += diff * diff;
    }

    return sum;

}

// biased estimate
// like in R too
float sd(vector<float> x) {
    float size = x.size();
    float diff = 1 / (size - 1);
    return sqrt(diff * S2(x));
}

// https://rdrr.io/cran/ks/src/R/normal.R
// TODO need to check
float psins1d(int r, float sigma) {

    float psins = 0;

    float r2 = float(r/2);

    if(r % 2 == 0) {
        psins = pow(-1, r2) * factorial(r) / (pow(2*sigma, r+1) * factorial(r/2) * sqrt(M_PI));
    }

    return psins;

}

vector<float> rep(float number, int amountOfRep) {
    vector<float> arr;

    for (int i = 0; i < amountOfRep; i++) {
        arr.push_back(number);
    }

    return arr;
}

vector<string> rep(string s, int amountOfRep) {
    vector<string> arr;

    for (int i = 0; i < amountOfRep; i++) {
        arr.push_back(s);
    }

    return arr;
}

vector<int> convertFromFloatVectorToIntVector(vector<float> v) {
    vector<int> out;
    for(int i = 0; i < v.size(); i++) {
        out.push_back(int(v[i]));
    }
    return out;
}

// https://rdrr.io/cran/ks/src/R/binning.R
vector<int> defaultGridsize(int d) {
    vector<int> gridsize;
    if(d == 1) {
        gridsize.push_back(401);
    }
    else if(d == 2) {
        gridsize = convertFromFloatVectorToIntVector(rep(151, d));
    }
    else if(d == 3) {
        gridsize = convertFromFloatVectorToIntVector(rep(51, d));
    }
    else if (d>=4){
        gridsize = convertFromFloatVectorToIntVector(rep(21, d));
    }
    else {
        cout << "Error: d < 1" << endl;
        exit(-1);
    }
    return gridsize;
}

// https://rdrr.io/cran/ks/src/R/binning.R
// TODO need to check
float linbinKs(vector<float> x, vector<float> gpoints, vector<float> w) {
    int n = x.size();
    int M = gpoints.size();
    if(w.size() == 0) {
        w = rep(1, n);
    }
    float a = gpoints[1];
    float b = gpoints[M];

    float xi = 0;
    // TODO
//    xi <- .C(C_massdist1d, x1=as.double(x[,1]), n=as.integer(n), a1=as.double(a), b1=as.double(b), M1=as.integer(M), weight=as.double(w), est=double(M))$est

    return xi;
}

// https://rdrr.io/cran/ks/src/R/binning.R
// TODO 1
// TODO need to check
/// h it is?
float binning1d(vector<float> x,
              vector<int> bgridsize,
              float h=-10000,
              float w=-10000,
              float xmin=-10000,
              float xmax=-10000,
              float H=-10000,
              float supp=3.7,
              char gridtype[] = "linear") {

    // return matrix like vector.T
    // like asmatrix in R
    float** matrix = toMatrix(x);

    int d = 1; // ncol
    int n = x.size(); // nrow
    vector<float> rangeX;
    vector<string> gridTypeVec;
    gridTypeVec.push_back("");

    if(w == -10000) {
        w = 1;
    }
    if(h == -10000) {
        h = 0;
    }
    if(H != -10000) {
        // TODO
    }
    if(bgridsize.size() == 0) {
        bgridsize = defaultGridsize(d);
    }
    if(xmin != -10000 && xmax != -10000) {
        rangeX.push_back(xmin);
        rangeX.push_back(xmax);
    }
    else if(xmin == -10000 || xmax == -10000) {
        // FIXME shape h is?
        float a = supp * h;
        rangeX.push_back(min(x) - a);
        rangeX.push_back(max(x) + a);
    }

    // FIXME not sure
    /// but 1d it's ok
    // a <- unlist(lapply(range.x,min))
    vector<float> a = rangeX;
    // b <- unlist(lapply(range.x,max))
    vector<float> b = rangeX;

    char defaultType[] = "linear";
    int k = strcmp(defaultType, gridtype);
    if(k != 0) {

    }

}

// https://rdrr.io/cran/ks/src/R/binning.R
// TODO need to check
// FIXME this will be for 2d
// TODO
float binning(vector<float> x,
                vector<float> h,
                vector<float> w,
                vector<int> bgridsize,
                vector<float> xmin,
                vector<float> xmax,
                float H = 0,
                float supp=3.7,
                string gridtype="linear") {

    // return matrix like vector.T
    // like asmatrix in R
    float** matrix = toMatrix(x);

    // TODO x should be not vector - matrix
    int d = 1; // ncol
    int n = x.size(); // nrow

    if(w.size() == 0) {
        w = rep(1, n);
    }
    if(h.size() == 0) {
        h = rep(0, d);
    }
    if(H != 0) {
        // TODO H not number it's bandwidth matrix
    }
    if(bgridsize.size() == 0) {
        bgridsize = defaultGridsize(d);
    }
    if(xmin.size() != 0 && xmax.size() == 0) {

    }

}

// https://rdrr.io/cran/ks/src/R/normal.R
// TODO need to check
float dnormDerivSum(vector<float> x, float sigma, float derivOrder, float binPar=0,
        float inc=1, bool binned=false, bool kfe=false) {
    // TODO
    float r = derivOrder;
    float n = x.size();
    float bin_par = binPar;

    if(binned) {
        if(binPar == 0) {
//            bin_par = binning(x, sigma, 4+r);
        }
    }
}

// https://rdrr.io/cran/ks/src/R/kfe.R
// TODO need to check
float kfe1d(vector<float> x, float g, float derivOrder, float binPar, float inc=1, bool binned=true) {
    float r = derivOrder;
    float n = x.size();

    float psir = dnormDerivSum(x, g, r, 1, binned, binPar, true);

    if(inc == 0) {
//        psir = (n^2*psir - n*dnorm.deriv(0, mu=0, sigma=g, deriv.order=r))/(n*(n-1));
    }

    return psir;
}

// https://rdrr.io/cran/ks/src/R/kfe.R
// TODO need to check
float hpiKfe(vector<float> x, float bgridsize, float nstage=2,
        bool binned=false, bool amise=false, float derivOrder=0) {

    float n = x.size();
    float d = 1;
    float r = derivOrder;
    float k = 2; // kernel order

    float Kr0 = dnormDeriv(0, 0, 1, r);

    float mu2K = 1;
    float psi2Hat = 0;

    if(nstage == 2) {
        float psi4Hat = psins1d(r+k+2, sd(x));
        float gamse2 = pow(factorial(r+2) * Kr0 / float(mu2K * psi4Hat * n), float(1) / float(r+k+3));
        // TODO
//        float psi2Hat = kfe1d(x=x, g=gamse2, deriv.order=r+k, inc=1, binned=binned);
    }

    else {
        psi2Hat = psins1d(r+k, sd(x));
    }

    float gamse = pow(factorial(r) * Kr0 / (-mu2K * psi2Hat * n), float(1) / float(r+k+1));

    return 0;

}

// TODO need to check
float kdeTest1d(vector<float> x1, vector<float> x2, int bgridsize) {
    int n1 = x1.size();
    int n2 = x2.size();
    int d = 1;

    float K0 = dnormDeriv(0, 0, 1, 1);

    float s1 = sd(x1);
    float s2 = sd(x2);

    // TODO

    return 0;
}

int main(int argc, char* argv[]) {
      vector<float> arr;

      arr.push_back(1);
      arr.push_back(2);
      arr.push_back(3);
      arr.push_back(4);
      arr.push_back(5);

}
#include <iostream>
#include <cmath>
#include <vector>
#include <string.h>
#include <map>

using namespace std;

/// based on https://www.mvstat.net/mvksa/mvksa.pdf

/// Matrix always is
/// (A, B)
/// (C, D)

/***
 * |H| - determinant of bandwidth matrix
 * H ^ (1/2) - squared matrix
 ***/

void print(int x) {
    cout << x << endl;
}

void print(float x) {
    cout << x << endl;
}

void print(string x) {
    cout << x << endl;
}

void print(vector<float> v) {
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "\n";
}

void print(vector<int> v) {
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "\n";
}

void print(vector<string> v) {
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

// is ok
// return matrix lines x columns
float** getMatrix(const int lines, const int columns) {
    float** matrix = new float * [lines];
    for (int i = 0; i < lines; i++) {
        matrix[i] = new float [columns];
    }
    return matrix;
}

// is ok
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

// is ok
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

// is ok
// return 2x2 matrix
float** get2dArray() {
    return getMatrix(2, 2);
}

// is ok
// return |H| - determinant of 2d matrix
float determinant2d(float matrix[2][2]) {
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

// is ok
// return |H| - determinant of 2d matrix
float determinant2d(float** matrix) {
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

/***
 * find sqrt(H)
 * https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix#:~:text=A%20square%20root%20of%20a,obtained%20by%20an%20explicit%20formula.
***/
// is ok
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
// is ok
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

// is ok
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

// is ok
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

// is ok
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

// is ok
void printTvector(float** vector, int len) {
    for (int i = 0; i < len; i++) {
        cout << vector[i][0] << endl;
    }
}

// is ok
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

// is ok
void multiplyMatrixToNumber(float** matrix, const int l, const int c, float number) {
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < c; j++) {
            matrix[i][j] /= number;
        }
    }
}

// TODO exp of matrix mb use it https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_exp
float normalKernelFunction2d(float* x, const int lenX) {

    float** resultMatrix = multiplyMatrix(t(x, lenX), toMatrixFromVector(x, lenX), 1, lenX, lenX, 1);
    multiplyMatrixToNumber(resultMatrix, lenX, lenX, 2);

//    float** Kx = pow(2 * M_PI, d / 2) * exp(resultMatrix);

}

// is ok
float** sumOfMatrix(float** matrix1, float** matrix2, const int l, const int c) {
    float** resultMatrix = getMatrix(l, c);

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < c; j++) {
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return resultMatrix;
}

// is ok
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
// is ok
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
// is ok
float** get2dEmatrix() {
    float** Ematrix = get2dArray();
    Ematrix[0][0] = 1;
    Ematrix[1][1] = 1;
    return Ematrix;
}

// is ok
// https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Normal.html
// https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Normal
float dnorm(float x, float mu=0, float sigma=1) {

    if (x == 0) {
        return 0.398942;
    }

    else if(x == 1) {
        return 0.241971;
    }

    else {
        float xMinusMu = x - mu;
        return (1 / sqrt(2 * M_PI) * sigma) * exp(-((xMinusMu * xMinusMu) / (2 * sigma * sigma)));
    }

}

// https://rdrr.io/cran/ks/src/R/normal.R
// is ok
float dnormDeriv(float x=0, float mu=0, float sigma=1, float derivOrder=0) {
    float r = derivOrder;
    float phi = dnorm(x, mu, sigma);

    float X = x - mu;
    float arg = x / sigma;

    float hmold0 = 1;
    float hmold1 = arg;
    float hmnew = 1;

    if (r == 1) {
        hmnew = hmold1;
    }
    else if (r >= 2) {
        for (int i = 0; i < r - 1; i++) {
            hmnew = arg * hmold1 - (i - 1) * hmold0;
            hmold0 = hmold1;
            hmold1 = hmnew;
        }
    }

    return pow(-1, r) * phi * hmnew / pow(sigma, r);
}

vector<float> dnormDeriv(vector<float> x, float mu=0, float sigma=1, float derivOrder=0) {
    vector<float> out; int len = x.size();

    for (int i = 0; i < len; i++) {
        out.push_back(dnormDeriv(x[i]));
    }

    return out;
}

// is ok
float mean(vector<float> x) {
    float sum = 0;
    for (int i = 0; i < x.size(); i++) {
        sum += x[i];
    }
    return sum / x.size();
}

// sample variance
// is ok
float S2(vector<float> x) {
    float m = mean(x);
    float sum = 0;

    for (int i = 0; i < x.size(); i++) {
        float diff = x[i] - m;
        sum += diff * diff;
    }

    return sum;

}

/// Standard deviation
// biased estimate like in R
// is ok
float sd(vector<float> x) {
    float size = x.size();
    float diff = 1 / (size - 1);
    return sqrt(diff * S2(x));
}

// https://rdrr.io/cran/ks/src/R/normal.R
// is ok
float psins1d(int r, float sigma) {

    float psins = 0;

    float r2 = float(r) / 2;

    if(r % 2 == 0) {
        psins = pow(-1, r2) * factorial(r) / (pow(2 * sigma, r + 1) * factorial(r / 2) * sqrt(M_PI));
    }

    return psins;

}

// is ok
vector<float> rep(float number, int amountOfRep) {
    vector<float> arr;

    for (int i = 0; i < amountOfRep; i++) {
        arr.push_back(number);
    }

    return arr;
}

// is ok
vector<string> rep(string s, int amountOfRep) {
    vector<string> arr;

    for (int i = 0; i < amountOfRep; i++) {
        arr.push_back(s);
    }

    return arr;
}

// is ok
vector<int> convertFromFloatVectorToIntVector(vector<float> v) {
    vector<int> out;
    for(int i = 0; i < v.size(); i++) {
        out.push_back(int(v[i]));
    }
    return out;
}

// is ok
// https://rdrr.io/cran/ks/src/R/binning.R
vector<float> defaultGridsize(int d) {
    vector<float> gridsize;
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
        cout << "Error in function defaultGridsize: d < 1" << endl;
        exit(-1);
    }
    return gridsize;
}

// TODO need to check
vector<float> massdist1d(vector<float> x1, int n, float a1, float b1, int M1,
                vector<float> weight) {

    vector<float> est;

    double fx1, wi, xdelta1, xpos1;
    int i, ix1, ixmax1, ixmin1, MM1;

    MM1 = M1;
    ixmin1 = 0;
    ixmax1 = MM1 - 2;
    xdelta1 = (b1 - a1) / (MM1 - 1);

    // set all est = 0
    for (i=0; i < MM1; i++) {
        est.push_back(0.0);
    }
    // assign linear binning weights
    // n - length of vector x1
    for (i = 0; i < n; i++) {
        if(!isinf(x1[i])) {
            xpos1 = (x1[i] - a1) / xdelta1;
            ix1 = floor(xpos1);
            fx1 = xpos1 - ix1;
            wi = weight[i];

            if(ixmin1 <= ix1 && ix1 <= ixmax1) {
                est[ix1] += wi * (1 - fx1);
                est[ix1 + 1] += wi * fx1;
            }
            else if(ix1 == -1) {
                est[0] += wi * fx1;
            }
            else if(ix1 == ixmax1 + 1) {
                est[ix1] += wi * (1 - fx1);
            }
        }
    }

    return est;
}

// https://rdrr.io/cran/ks/src/R/binning.R
// TODO need to check
vector<float> linbinKs(vector<float> x, vector<float> gpoints, vector<float> w) {

    // n - length of vector
    int n = x.size();
    int M = gpoints.size();

    if(w.size() == 0) {
        w = rep(1.0, n);
    }

    float a = gpoints[0];
    float b = gpoints[M - 1];

    return massdist1d(x, n, a, b, M, w);
}

// is ok
vector<float> seq(float start, float end, float size=-1) {
    vector<float> v;

    int roundedSize = int(size);

    if(size != roundedSize) {
        roundedSize++;
    }

    if (size == -1) {
        roundedSize = abs(start) + abs(end) + 1;
    }

    if(roundedSize == 1) {
        v.push_back(start);
        return v;
    }
    else if(roundedSize == 2) {
        v.push_back(start);
        v.push_back(end);
        return v;
    }
    else {
        float delitemer = (end - start) / (roundedSize - 1);
        float number = start;
        while(number <= end) {
            v.push_back(number);
            number += delitemer;
        }
    }

    return v;
}

// https://rdrr.io/cran/ks/src/R/binning.R
// TODO need to check
map<string, vector<float>> binning1d(vector<float> x,
                vector<int> bgridsize,
                vector<float> w,
                char gridtype[],
                float h=-10000,
                float xmin=-10000,
                float xmax=-10000,
                float H=-10000,
                float supp=3.7) {

    int d = 1; // ncol
    int n = x.size(); // nrow

    vector<float> rangeX;
    vector<float> gpoints;

    if(w.size() == 0) {
        w = rep(1.0, n);
    }

    if(h == -10000) {
        h = 0;
    }

    if(H != -10000) {
        h = sqrt(H);
    }

    if(bgridsize.size() == 0) {
        bgridsize = defaultGridsize(d);
    }

    if(xmin != -10000 && xmax != -10000) {
        rangeX.push_back(xmin);
        rangeX.push_back(xmax);
    }

    else if(xmin == -10000 || xmax == -10000) {
        float supph = supp * h;
        rangeX.push_back(min(x) - supph);
        rangeX.push_back(max(x) + supph);
    }

    float minX = min(rangeX);
    float maxX = max(rangeX);

    char linear[] = "linear";
    char sqrt[] = "sqrt";
    char quantile[] = "quantile";
    char log[] = "log";

    if(strcmp(gridtype, linear) == 0) {
        gpoints = seq(minX, maxX, float(bgridsize[0]));
    }
    else if(strcmp(gridtype, log) == 0) {
        gpoints = seq(exp(minX), exp(maxX), float(bgridsize[0]));
    }

    vector<float> counts = linbinKs(x, gpoints, w);

    map<string, vector<float>> m;
    m["counts"] = counts;
    m["eval_points"] = gpoints;
    m["w"] = w;

    return m;

}

// is ok
float sum(vector<float> v) {
    float summ = 0;
    for (int i = 0; i < v.size(); i++) {
        summ += v[i];
    }
    return summ;
}

// is ok
vector<float> div(vector<float> x, float number) {
    vector<float> out; float len = x.size();

    for (int i = 0; i < len; i++) {
        out.push_back(x[i] / number);
    }

    return out;
}

// is ok
vector<float> mult(vector<float> x, float number) {
    vector<float> out; float len = x.size();

    for (int i = 0; i < len; i++) {
        out.push_back(x[i] * number);
    }

    return out;
}

// is ok
int defaultBflag(float d, float n) {
    int thr;
    if (d == 1) {
        thr = 1;
    }
    else if (d == 2) {
        thr = 500;
    }
    else if (d > 2) {
        thr = 1000;
    }
    return n > thr ? 1 : 0;

}

// https://rdrr.io/cran/ks/src/R/prelim.R
// only for 1d
// TODO 1
float ksdefaults(vector<float> x, vector<float> w, float bgridsize, float gridsize, int binned=2) {
    float d = 1;
    float n = x.size();

    if(w.size() == 0) {
        w = rep(1, n);
    }
    else {
        print("Not implemented");
        exit(-1);
    }

    if(binned == 2) {
        binned = defaultBflag(d, n);
    }



//## default grid sizes
//if (missing(bgridsize))
//{
//if (missing(gridsize)) bgridsize <- default.bgridsize(d)
//else bgridsize <- gridsize
//}
//if (missing(gridsize)) gridsize <- default.gridsize(d)
//if (length(gridsize)==1) gridsize <- rep(gridsize, d)
//if (length(bgridsize)==1) bgridsize <- rep(bgridsize, d)
//
//return(list(d=d, n=n, w=w, binned=binned, bgridsize=bgridsize, gridsize=gridsize))
}

// https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/fft
// TODO 2
vector<float> fft(vector<float> x) {

}

// TODO 1
vector<float> replace(vector<float> x1, vector<float> x2, int start, int end) {
    vector<float> out; int len = x1.size();

    // copy
    for (int i = 0; i < len; i++) {
        out.push_back(x1[i]);
    }

    int j = 0;
    for (int i = start; i <= end; i++) {
        out[i] = x2[j++];
    }

}

// TODO 3
float symconv1d(vector<float> keval, vector<float> gcounts) {

    float M = gcounts.size();
    float N = keval.size();
    float L = (N + 1) / 2;

    // Smallest powers of 2 >= M + N
    float P = pow(2, log2(M + N));

    // Zero-padded version of keval an gcounts
    vector<float> kevalZeropad = rep(0, P);
    vector<float> gcountsZeropad = rep(0, P);
//    keval.zeropad[0:N-1] <- keval;
//    gcounts.zeropad[L:(L+M-1)] <- gcounts;
//
//    ## FFTs
//        K <- fft(keval.zeropad)
//C <- fft(gcounts.zeropad)
//
//## Invert element-wise product of FFTs and truncate and normalise it
//        symconv.val <- Re(fft(K*C, inverse=TRUE)/P)[N:(N+M-1)]
        return 0;
}

// TODO 4
float kddeBinned1d1d(float h, float derivOrder, map<string, vector<float>> binpar) {

    vector<float> evalPoints;

    float r = derivOrder;
    float n = sum(evalPoints);
    float a = min(evalPoints);
    float b = max(evalPoints);
    float M = evalPoints.size();
    float L = min((4 + r) * h * (M - 1) / (b - a), M - 1);
    float delta = (b - a) / (M - 1);
    float N = 2 * L - 1;

    vector<float> grid1 = seq(-(L - 1), L - 1);
    vector<float> keval = div(dnormDeriv(mult(grid1, delta), 0, h, derivOrder=r), n);

}

// TODO 5
// https://rdrr.io/cran/ks/src/R/kdde.R
float kddeBinned1d(vector<float> x,
        vector<float> bgridsize, float xmin, float xmax, map<string, vector<float>> binpar,
        vector<float> w, int derivIndex,
        float H=-10000, float h=-10000, float derivOrder=0,
        bool derivVec=true) {

    print("function kddeBinned1d called");

    float d;

    // if missing binpar no needed yet
    // else

    vector<float> evalpoins = binpar["eval_points"];
    unsigned int evalpoinsSize = evalpoins.size();

    if (evalpoinsSize == 0) {
        cout << "Error in function kddeBinned1d, evalpoins.size() in binpar = 0\n";
        exit(-1);
    }

    else if(evalpoinsSize == 1) {
        d = 1;
        bgridsize.push_back(evalpoinsSize);
    }

    else {
        d = evalpoinsSize;
        // TODO for 2d
//        bgridsize = sapply(evalpoins, length);
    }

    w = binpar["w"];

    if(d == 1) {
        // if H is missing
        if(H == -10000) {
            H = h * h;
        }
        else {
            h = sqrt(H);
        }
    }

    if (d == 1) {
//        fhat = kdde.binned.1d(h=h, deriv.order=r, bin.par=bin.par)
//        evalpoints = fhat["eval_points"];
//        est = fhat["estimate"];
    }

//    if (d == 1) {
//        if (r == 0) {
//            fhat = list(x=x, eval.points=unlist(eval.points), estimate=est, h=h, H=h^2, gridtype="linear", gridded=TRUE, binned=TRUE, names=NULL, w=w)
//        }
//        else {
//            fhat = list(x=x, eval.points=unlist(eval.points), estimate=est, h=h, H=h^2, gridtype="linear", gridded=TRUE, binned=TRUE, names=NULL, w=w, deriv.order=r, deriv.ind=r)
//        }
//    }

}

vector<float> minus(float number, vector<float> v) {
    vector<float> out;

    for (int i = 0; i < v.size(); i++) {
        out.push_back(number - v[i]);
    }

    return out;
}

// https://rdrr.io/cran/ks/src/R/normal.R
// TODO need to check
float dnormDerivSum(vector<float> x, float sigma, float derivOrder,
        float inc=1, bool binned=false, bool kfe=false) {

    float sumval;

    float r = derivOrder;
    float n = x.size();
    map<string, vector<float>> bin_par;

    // no need
    if(binned) {
        cout << "Error in dnormDerivSum if binned not implemened\n";
        exit(-1);
    }

    else {
        sumval = 0;
        for (int i = 0; i < n; i++) {
            sumval += sum(dnormDeriv(x=minus(x[i], x), mu=0, sigma=sigma, derivOrder=r));
        }
        if (inc == 0) {
            sumval -= n * derivOrder(0);
        }
    }

    if (kfe) {
        if (inc == 1) {
            sumval /= n * n;
        }
        else {
            sumval /= n * (n - 1);
        }
    }

    return sumval;
}

// https://rdrr.io/cran/ks/src/R/kfe.R
// TODO need to check
// binned = false
float kfe1d(vector<float> x, float g, float derivOrder, float binPar, float inc=1, bool binned=true) {
    float n = x.size();

    float psir = 0;

     float psir = dnormDerivSum(x, g, derivOrder, 1, binned, binPar, true);

    if(inc == 0) {
        psir = pow(n, 2 * psir - n * dnormDeriv(0, 0, g, derivOrder)) / (n * (n - 1));
    }

    return psir;
}

float hpi(vector<float> x, vector<float> bgridsize, float nstage=2, binned=true, float derivOrder=0) {
    float d = 1; float h; float n; float r; float m2; float mr; float psi2r4hat;

    if (derivOrder == 0) {
        cout << "Error in hpi, dpik not implemented\n";
        exit(-1);
//        h = dpik(x=x, level=nstage, gridsize=bgridsize);
    }
    else {
        n = x.size();
        r = derivOrder;
        float K2r4 = dnormDeriv(x=0, mu=0, sigma=1, derivOrder=2*r+4);
        float K2r6 = dnormDeriv(x=0, mu=0, sigma=1, derivOrder=2*r+6);
        m2 = 1;
        mr = psins1d(r=2*r, sigma=1);
    }

    if(nstage == 2) {
        float psi2r8hat = psins1d(r=2*r+8, sigma=sd(x));
        float gamse2r6 = pow((2 * K2r6 / (-m2 * psi2r8hat * n)), (float(1) / (2 * r + 9)));
        float psi2r6hat = kfe1d(x, gamse2r6, 2*r+6, 1, binned);
        float gamse2r4 = pow((2 * K2r4 / (-m2 * psi2r6hat * n)), (float(1) / (2 * r + 7)));
        psi2r4hat = kfe1d(x, gamse2r4, 2*r+4, 1, binned);
    }
    else {
        float psi2r6hat = psins1d(r=2*r+6, sigma=sd(x));
        float gamse2r4 = pow((2 * K2r4 / (-m2 * psi2r6hat * n)), (float(1) / (2 * r + 7)));
        psi2r4hat = kfe1d(x, gamse2r4, 2*r+4, 1, binned);
    }

    return pow(((2 * r + 1) * mr / (m2 * m2 * psi2r4hat * n)), float(1) / (2 * r + 5));
}

float hpi(vector<float> x, float nstage=2, bool binned=true, float derivOrder=0) {
    vector<float> bgridsize = defaultGridsize(d);
    return hpi(x, bgridsize, nstage, binned, derivOrder);
}

// https://rdrr.io/cran/ks/src/R/kfe.R
// TODO need to check
// binned = false
float hpiKfe(vector<float> x, float nstage=2, bool binned=false, float derivOrder=0) {

    float n = x.size();
    float d = 1;
    float k = 2; // kernel order

    float Kr0 = dnormDeriv(0, 0, 1, derivOrder);

    float mu2K = 1;
    float psi2Hat = 0;

    if(nstage == 2) {
        float psi4Hat = psins1d(derivOrder + k + 2, sd(x));
        float gamse2 = pow(factorial(derivOrder + 2) * Kr0 / float(mu2K * psi4Hat * n), float(1) / float(derivOrder + k + 3));
        float psi2Hat = kfe1d(x, gamse2, derivOrder + k, 1, binned);
    }

    else {
        psi2Hat = psins1d(derivOrder+k, sd(x));
    }

    return pow(factorial(derivOrder) * Kr0 / (-mu2K * psi2Hat * n), float(1) / float(derivOrder+k+1));

}

// is ok
float max(float a, float b) {
    return a > b ? a : b;
}

// is ok
vector<float> seq2(float start, float end, float by) {
    vector<float> v;
    float s = start;
    while(s <= end) {
        v.push_back(s);
        s += by;
    }

    return v;
}

// TODO need to check
vector<float> blockIndices(float nx, float ny, float d,
        float npergroup=-10000, float r=0, bool diff=false, float blockLimit=1e6) {

    print("function blockIndices called");
    cout << "Args: nx = " << nx << ", ny = " << ny << ", d = " << d;
    cout << ", npergroup = " << npergroup << ", r = " << r;
    cout << ", diff = " << diff << ", blockLimit = " << blockLimit << endl;

    if(npergroup == -10000) {
        if(diff) {
            npergroup = max(int(blockLimit) / int(nx * pow(d, r)), 1);
        }
        else {
            npergroup = max(int(blockLimit) / int(nx), 1);
        }
    }

    vector<float> nseq = seq2(1, ny, npergroup);

    if(nseq[0] <= ny) {
        nseq.push_back(ny + 1);
    }

    if(nseq.size() == 1) {
        nseq.push_back(1);
        nseq.push_back(ny + 1);
    }

    return nseq;

}

// is ok
vector<float> squares(vector<float> v) {
    vector<float> out;
    for (int i = 0; i < v.size(); i++) {
        float a = v[i];
        out.push_back(a * a);
    }
    return out;
}

// is ok
vector<float> get(vector<float> v, int start, int end) {
    vector<float> out;

    if(end >= v.size()) {
        cout << "Error in function get: index of bound exception" << endl;
        exit(-1);
    }

    for (int i = start; i <= end; i++) {
        out.push_back(v[i]);
    }

    return out;
}


// is ok
float hns(vector<float> x, float derivOrder=0) {

    int n = x.size();
    int d = 1;
    float dplus2derivOrder = d + 2 * derivOrder;

    return pow(4 / (n * (dplus2derivOrder + 2)), 1 / (dplus2derivOrder + 4)) * sd(x);
}

// TODO 1
// derivOreder = 0 оно не переопределяется для 1d
float qr1d(vector<float> x, vector<float> y,
        float sigma, float derivOrder=0, float inc=1) {

    print("function qr1d");
    cout << "Args: sigma = " << sigma << ", derivOrder = " << derivOrder << ", inc = " << inc << endl;

    int d = 1;
    float r = derivOrder / 2;

    cout << "r = " << r << endl;

    float nx = x.size();
    float ny = y.size();
    float g = sigma;

    cout << "nx = " << nx << ", ny = " << ny << ", g = sigma = " << sigma << endl;

    vector<float> nseq = blockIndices(nx, ny, 1, r=0);

    float eta = 0;

    vector<float> a = squares(x);

    if(r == 1) {
        for (int i = 0; i < nseq.size() - 1; i++) {
            float nytemp = nseq[i + 1] - nseq[i];
            vector<float> ytemp = get(y, int(nseq[i]), int(nseq[i + 1] - 1));
            vector<float> aytemp = squares(ytemp);
//            float M = a %*% t(rep(1, nytemp)) + rep(1, nx) %*% t(aytemp) - 2 * (x %*% t(ytemp));
//            float em2 = exp(-M / (2 * g * g));
//            eta = eta + pow(2 * M_PI, -float(d) / float(2)) * float(1) / float(g) * sum(em2);
        }
    }
    // no need
    else if(r > 0) {
        cout << "Error in function qr1d, r > 0 not implemented" << endl;
        exit(-1);
    }
    else {
        cout << "Error in function qr1d, r = " << r << " it's should't be < 0" << endl;
        exit(-1);
    }


}

// TODO 4
// binned = false
float kdeTest1d(vector<float> x1, vector<float> x2,
        vector<float> bgridsize, vector<float> gridsize, bool binned) {

    int n1 = x1.size();
    int n2 = x2.size();
    int d = 1;

    float K0 = dnormDeriv(0, 0, 1, 1);

    float s1 = sd(x1);
    float s2 = sd(x2);

    float h1 = hpiKfe(x1);
    float h2 = hpiKfe(x2);

    float psi1 = qr1d(x1, x1, sigma=h1);
    float psi2 = qr1d(x2, x2, sigma=h2);

    float h1r1 = hns(x1, derivOorder=1);
    // fhat1r1 = predict(kdde(x=x1, h=h1.r1, deriv.order=1), x=mean(x1))
    // varfhat1 = fhat1.r1^2*s1^2

    float psi12 = qr1d(x1, x2, sigma=h1);

    float h2r1 = hns(x2, derivOrder=1);
    // fhat2r1 = predict(kdde(x=x2, h=h2.r1, deriv.order=1), x=mean(x2))
    // varfhat2 = fhat2.r1^2*s2^2

    float psi21 = qr1d(x2, x1, sigma=h2);

    float That = drop(psi1 + psi2 - (psi12 + psi21));
    // muT.hat <- ((n1*h1)^(-1) + (n2*h2)^(-1))*K0
    // varT.hat <- 3*(n1*var.fhat1 + n2*var.fhat2)/(n1+n2) *(1/n1+1/n2)
    //zstat <- (T.hat-muT.hat)/sqrt(varT.hat)
    // pval <- 1-pnorm(zstat)
    // if (length(pval==0)>0) pval[pval==0] <- pnorm(-abs(zstat[pval==0]))

    return 0;
}

// https://rdrr.io/cran/ks/src/R/binning.R
// is ok
vector<float> defaultBgridsize(int d) {

    if (d == 1) {
        return rep(401, d);
    }

    if (d == 2) {
        return rep(151, d);
    }

}

// binned = false
map<string, vector<float>> ksDefaults(vector<float> x1, bool binned) {
    int d = 1;
    float n = x1.size();
    vector<float> w = rep(1, n);
    vector<float> bgridsize = defaultBgridsize(d);
    vector<float> gridsize = defaultGridsize(d);

    map<string, vector<float>> m;
    m["w"] = w;
    m["bgridsize"] = bgridsize;
    m["gridsize"] = gridsize;

    return m;

}

float kdeTest(vector<float> x1, vector<float> x2, bool binned=false) {
    map<string, vector<float>> ksd = ksDefaults(x=x1, binned=binned);
    int d = 1; int n = x1.size();

    vector<float> bgridsize = ksd["bgridsize"];
    vector<float> gridsize = ksd["gridsize"];

    return kdeTest1d(x1, x2, bgridsize, gridsize, binned);
}

int main(int argc, char* argv[]) {
      vector<float> arr;

      arr.push_back(1);
      arr.push_back(2);
      arr.push_back(3);
      arr.push_back(4);
      arr.push_back(5);

      print(dnormDeriv(1));
}
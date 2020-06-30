#include <iostream>
#include <cmath>

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

// return matrix lines x columns
float** getMatrix(const int lines, const int columns) {
    cout << lines << " " << columns << endl;
    float** matrix = new float * [lines];
    for (int i = 0; i < lines; i++) {
        matrix[i] = new float [columns];
    }
    return matrix;
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
// TODO 4 matrix instead 1, which should be chosen?
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

/***
 * https://www.mathsisfun.com/algebra/matrix-inverse.html
 * @param matrix
 * @return matrix ^-1
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
    float ** tVector = new float* [len];

    for (int i = 0; i < len; i++) {
        tVector[i] = new float [1];
    }

    for (int i = 0; i < len; i++) {
        tVector[i][0] = vector[i];
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
    multiplyMatrixToNumber(resultMatrix, lenX, lenX);

    float** Kx = pow(2 * M_PI, d / 2) * exp(resultMatrix);

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

float** matrixExp(float** matrix, const int dim) {

    float** resultMatrix = getMatrix(dim, dim);

    for (int i = 0; i < N; i++) {

    }

}

int main(int argc, char* argv[]) {
    // 2d matrix to bandwidth
    float H[2][2];

//    float** arr = get2dArray();
//    arr[0][0] = 4.0;
//    arr[0][1] = 7.0;
//    arr[1][0] = 2.0;
//    arr[1][1] = 6.0;
//    print2dArr(arr);
//
//    float** iMatrix = getInverse2dMatrix(arr);
//    print2dArr(iMatrix);

//    float** testSqrtMatrix = get2dArray();
//    testSqrtMatrix[0][0] = 2;
//    testSqrtMatrix[0][1] = 2;
//    testSqrtMatrix[1][0] = 3;
//    testSqrtMatrix[1][1] = 4;
//    print2dArr(testSqrtMatrix);
//
//    float** squaredMatrix = getSquareRoot2dMatrix(testSqrtMatrix);
//    print2dArr(squaredMatrix);

//    const int len = 5;
//    float* v = new float[len];
//    v[0] = 1;
//    v[1] = 2;
//    v[2] = 3;
//    v[3] = 4;
//    v[4] = 5;
//    printTvector(t(v, len), len);
}
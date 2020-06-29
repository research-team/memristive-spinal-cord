#include <iostream>
#include <cmath>

using namespace std;

/// based on https://www.mvstat.net/mvksa/mvksa.pdf

/// Matrix always is
/// (A, B)
/// (C, D)

/***
 * |H| - determinant of bandwidth matrix
 * H ^ (1/2) - squared matrix
 ***/

float **get2dArray() {
    float ** array = new float * [2];
    for (int i = 0; i < 2; i++) {
        array[i] = new float [2];
    }
    return array;
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
    float ** squareRoot2dMatrix = get2dArray();

    float determinantOfH = determinant2d(matrix);
    cout << "determinant = " << determinantOfH << endl;

    if (determinantOfH < 0) {
        cout << "Error: determinant of matrix < 0" << endl;
        exit(-1);
    }

    float s1 = sqrt(determinantOfH);
    float s2 = -s1;
    cout << "s1 = " << s1 << endl;
    cout << "s2 = " << s2 << endl;

    // A + D
    float r = matrix[0][0] + matrix[1][1];
    cout << "A + D = " << r << endl;

    float sum = r + 2 * s1;
    cout << "sum = " << sum << endl;

    if (sum < 0) {
        cout << "Error: sum" << endl;
        exit(-1);
    }

    float t = sqrt(sum);
    cout << "t = " << t << endl;

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

void print2dMatrix(float matrix[2][2]) {
    cout << "-----------\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "-----------\n";
}

void print2dArr(float **matrix) {
    cout << "-----------\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "-----------\n";
}

/***
 * https://www.mathsisfun.com/algebra/matrix-inverse.html
 * @param matrix
 * @return matrix ^-1
 ***/
float **getInverse2dMatrix(float **matrix) {
    float ** inverse2dMatrix = get2dArray();

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

// .T
float **t(float* vector, const int len) {
    float ** tVector = new float * [len];

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

float **xTx(float *x) {

}

int main(int argc, char* argv[]) {
    // amount of dimensional
    int d = 2;
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

    const int len = 5;
    float* v = new float[len];
    v[0] = 1;
    v[1] = 2;
    v[2] = 3;
    v[3] = 4;
    v[4] = 5;
    printTvector(t(v, len), len);
}
import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def get_spectral_radius(matrix):
    eigan_vals = np.linalg.eigvals(matrix)
    rho = max([math.fabs(val) for val in eigan_vals])

    return rho

def jacobi_iteration_matrix(matrix):
    d = np.diag(np.diag(matrix))
    eye = np.identity(len(matrix))
    da = np.linalg.solve(d, matrix)

    return np.subtract(eye, da)


def GS_iteration_matrix(matrix):
    l = np.tril(matrix)
    eye = np.identity(len(matrix))
    la = np.linalg.solve(l, matrix)

    return np.subtract(eye, la)

def lud_factorization(matrix):
    l = np.zeros(matrix.shape)
    u = np.zeros(matrix.shape)
    d = np.zeros(matrix.shape)

    for rowIndex, row in enumerate(matrix):
        for colIndex, col in enumerate(row):
            if rowIndex < colIndex:
                l[rowIndex][colIndex] = matrix[rowIndex][colIndex]
            elif rowIndex == colIndex:
                d[rowIndex][colIndex] = matrix[rowIndex][colIndex]
            else:
                u[rowIndex][colIndex] = matrix[rowIndex][colIndex]

    return l, u, d


def GS_step (a_matrix, b_vector, previous_step, ld_inv):
    ax_prev = np.dot(a_matrix, previous_step)
    b_minus_ax = np.subtract(b_vector, ax_prev)

    return np.add(previous_step, np.dot(ld_inv, b_minus_ax))



def l2_norm(vector):
    return math.sqrt(sum([math.pow(item, 2) for item in vector]))


def get_residue_norm(a_matrix, b_vector, step_result):
    ax = np.dot(a_matrix, step_result)
    ax_minus_b = np.subtract(ax, b_vector)

    return l2_norm(ax_minus_b)


def GS (a_matrix, b_vector, initial_guess, max_residue):
    l, u, d = lud_factorization(a_matrix)
    ld_inv = np.linalg.inv(np.add(l, d))

    step_results = [initial_guess]
    residues = [get_residue_norm(a_matrix, b_vector, initial_guess)]

    while residues[-1] > max_residue:
        nextIteration = GS_step(a_matrix, b_vector, step_results[-1], ld_inv)

        step_results.append(nextIteration)
        residues.append(get_residue_norm(a_matrix, b_vector, nextIteration))

    return step_results, residues


def get_error_norms(a_matrix, b_vector, results_steps):
    actual_results = np.linalg.solve(a_matrix, b_vector)
    error_values = [np.subtract(result_step, actual_results)
            for result_step in results_steps]

    return [l2_norm(error) for error in error_values]

def residue_ratio(residues):
    ratios = []
    for i, val in enumerate(residues[:-1]):
        ratios.append(residues[i + 1] / residues[i])

    return ratios



def plot(step_results, residues, errors):
    f, axarr = plt.subplots(4, 1)

    axarr[0].set_title("Residue norms")
    axarr[0].semilogy(residues)

    axarr[1].set_title("Error norms")
    axarr[1].semilogy(errors)

    axarr[2].set_title("Results")
    axarr[2].plot([tuple[0] for tuple in step_results], color='red', label="x1")
    axarr[2].plot([tuple[1] for tuple in step_results], color='yellow', label="x2")
    axarr[2].plot([tuple[2] for tuple in step_results], color='green', label="x3")
    axarr[2].plot([tuple[3] for tuple in step_results], color='blue', label="x4")
    axarr[2].plot([tuple[4] for tuple in step_results], color='purple', label="x5")

    axarr[3].set_title("Residue ratio")
    axarr[3].plot(residue_ratio(residues), color='red')

    plt.show()



a = np.array([[-5, 0.2, 0.2, 0.2, 0.2],
              [0.2, -5, 0.2, 0.2, 0.2],
              [0.2, 0.2, -5, 0.2, 0.2],
              [0.2, 0.2, 0.2, -5, 0.2],
              [0.2, 0.2, 0.2, 0.2, -5]
              ])
print("Jacobi spectral radius: " + str(get_spectral_radius(jacobi_iteration_matrix(a))))
print("GS spectral radius: " + str(get_spectral_radius(GS_iteration_matrix(a))))

b = np.array([1, 2, 3, 4, 5])
initial_guess = np.array([1, 1, 1, 1, 1])

results, residues = GS(a, b, initial_guess, 0.0000001)
errors = get_error_norms(a, b, results)
plot(results, residues, errors)

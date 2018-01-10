import pylab as pylab


class ResultsPlotter:
    def __init__(self, rows_number):
        self._rows_number = rows_number
        self._cols_number = rows_number
        self._plot_index = 1

    def reset(self):
        pylab.figure()
        pylab.title('Layer1 results')

    def show(self):
        pylab.show()

    def subplot(self, flexor, extensor, title):
        if self._plot_index >= self._rows_number:
            raise ValueError("Too many subplots!")
        pylab.subplot(self._rows_number, self._cols_number, self._plot_index)
        self._plot_index += 1

        pylab.plot(flexor['time'], flexor['value'], 'r', label='flexor')
        pylab.plot(extensor['time'], extensor['value'], 'b', label='extensor')

        pylab.ylabel(title)
        pylab.legend()

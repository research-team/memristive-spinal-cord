import pylab as pylab


class ResultsPlotter:
    def __init__(self, rows_number, title):
        self._rows_number = rows_number
        self._cols_number = 1
        self._plot_index = 1
        self._title = title

    def reset(self):
        pylab.figure()
        pylab.title(self._title)

    def show(self):
        pylab.show()

    def subplot(self, flexor, extensor, title):
        if self._plot_index > self._rows_number:
            raise ValueError("Too many subplots!")
        pylab.subplot(self._rows_number, self._cols_number, self._plot_index)
        self._plot_index += 1

        pylab.plot(flexor.keys(), flexor.values(), 'r.', label='flexor')
        pylab.plot(extensor.keys(), extensor.values(), 'b-.', label='extensor')

        pylab.ylabel(title)
        pylab.legend()

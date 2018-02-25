import os


with open('ees_tests.md', 'w') as report:
    report.write('### Examples of EES action  \n')
    for file in sorted(os.listdir('test_figures')):
        report.write('![](test_figures/{})  \n'.format(file))

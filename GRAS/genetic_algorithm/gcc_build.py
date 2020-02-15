import logging
import subprocess

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

class Build:
    buildname = "a.out"
    script_place = "/home/yuliya/Desktop/ga_v3"
    gcc = "/usr/bin/g++"
    buildfile = "human_pattern_NODAC.cpp"

    @staticmethod
    def create_build_string(individual):
        return ' ' + ' '.join(map(str, individual.gen))

    @staticmethod
    def compile():
        cmd_build = f"{Build.gcc} -fopenmp {Build.script_place}/{Build.buildfile}"
        logger.info(f"Execute: {cmd_build}")
        process = subprocess.Popen(cmd_build, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()

        for output in str(out.decode("UTF-8")).split("\n"):
            logger.info(output)
        for error in str(err.decode("UTF-8")).split("\n"):
            logger.info(error)

    @staticmethod
    def build(individual):
        cmd_run = f"./{Build.buildname}" + ' ' + Build.create_build_string(individual)

        logger.info(f"Execute: {cmd_run}")
        process = subprocess.Popen(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()

        for output in str(out.decode("UTF-8")).split("\n"):
            logger.info(output)
        for error in str(err.decode("UTF-8")).split("\n"):
            logger.info(error)
from colorama import init, Fore, Back, Style


class Console:
    @staticmethod
    def info(msg):
        init(autoreset=True)
        print(Fore.LIGHTBLUE_EX + "[INFO]:{} ".format(msg))

    @staticmethod
    def error(msg):
        init(autoreset=True)
        print(Fore.RED+"[ERR]:{}".format(msg))

    @staticmethod
    def ok(msg):
        init(autoreset=True)
        print(Fore.GREEN+"[OK]:{}".format(msg))

    @staticmethod
    def debug(msg):
        init(autoreset=True)
        print(Fore.MAGENTA+"[DEBUG]:{}".format(msg))


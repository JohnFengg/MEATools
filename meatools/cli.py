import argparse
from .subcomands import mea_proccess

def main():
    parser=argparse.ArgumentParser(prog="mea", description="MEATOOLs CLI Toolkit")
    subparsers=parser.add_subparsers(dest="command")

    lte_parser=subparsers.add_parser("ttseq",help='Run test sequence analysis')
    lte_parser.set_defaults(func=mea_proccess.run_test_sequence)

    lte_parser=subparsers.add_parser("otr",help='Run OTR/Impedance analysis')
    lte_parser.set_defaults(func=mea_proccess.run_otr)

    lte_parser=subparsers.add_parser("ecsa",help='Run ECSA analysis')
    lte_parser.set_defaults(func=mea_proccess.run_ecsa)

    lte_parser=subparsers.add_parser("ecsadry",help='Run ECSA-dry analysis')
    lte_parser.set_defaults(func=mea_proccess.run_ecsa_dry)

    lte_parser=subparsers.add_parser("lsv",help='Run LSV analysis')
    lte_parser.set_defaults(func=mea_proccess.run_lsv)

    args=parser.parse_args()
    if hasattr(args,'func'):
        args.func(args)
    else:
        parser.print_help()
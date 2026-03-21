"""Entry point for IP-Native Venture Engine."""

import os

from .screens import (
    main_menu,
    screen_select_domain_and_years,
    screen_confirm_and_run,
    screen_show_results,
    clear_screen,
    pause,
)


def run_app() -> None:
    while True:
        choice = main_menu()

        if choice == 1:
            context = screen_select_domain_and_years()
            if not context:
                continue
            results = screen_confirm_and_run(context)
            screen_show_results(results, context)

        elif choice == 2:
            from .domains import DOMAINS
            clear_screen()
            print("\nCurrent domains:\n")
            for d in DOMAINS:
                print(f"  {d['id']:>2})  {d['label']}")
            pause()

        elif choice == 3:
            clear_screen()
            print("Goodbye.\n")
            break


if __name__ == "__main__":
    run_app()

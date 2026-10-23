"""
Entry point
"""

from logistics_agent.app_logistics import logisticsbot

# ===

def main(argv: list[str] | None = None) -> None:
    logisticsbot().run(debug=False, port=8080)

if __name__ == '__main__':
    main()

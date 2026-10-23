"""
Entry point
"""

from logistics_agent.app_logistics import logisticsbot

# ===

app = logisticsbot()

def main(_argv: list[str] | None = None) -> None:
    app.run(debug=False, port=8080)

if __name__ == '__main__':
    main()

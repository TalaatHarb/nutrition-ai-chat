import ast
import math
import operator
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        return BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError


def calculator(expression: str) -> str:
    if not isinstance(expression, str) or not re.fullmatch(r"[0-9+*/(). -]{1,120}", expression):
        return "Error: use up to 120 characters with numbers, +, -, *, /, and parentheses."
    try:
        result = _evaluate(ast.parse(expression.strip(), mode="eval").body)
        if math.isfinite(result):
            return str(result)
    except (SyntaxError, ValueError, ArithmeticError):
        pass
    return "Error: invalid expression or result out of range (check for division by zero)."


def current_datetime(timezone: str) -> str:
    if not isinstance(timezone, str) or not re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", timezone):
        return "Error: provide an IANA time zone such as America/New_York or UTC."
    try:
        return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
    except ZoneInfoNotFoundError:
        return "Error: unknown time zone; use an IANA time zone such as America/New_York or UTC."


def shift_datetime(start: str, days: int = 0, minutes: int = 0) -> str:
    if type(days) is not int or type(minutes) is not int:
        return "Error: days and minutes must be integers."
    if not isinstance(start, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:T[0-9]{2}:[0-9]{2})?", start
    ):
        return "Error: use YYYY-MM-DD or YYYY-MM-DDTHH:MM (local time)."
    if len(start) == 10 and minutes:
        return "Error: include a start time when shifting by minutes."
    try:
        offset = timedelta(days=days, minutes=minutes)
        if len(start) == 10:
            shifted = date.fromisoformat(start) + offset
            return shifted.isoformat()
        shifted = datetime.fromisoformat(start) + offset
        return shifted.isoformat(timespec="minutes")
    except (ValueError, OverflowError):
        return "Error: invalid start date/time or offset out of range."


def _tool(function, description, required, **properties):
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool(
        calculator,
        "Evaluate basic arithmetic for portions and nutrient estimates.",
        ["expression"],
        expression={"type": "string", "description": "Numbers, +, -, *, / and (), e.g. 15 * 12 + 3"},
    ),
    _tool(
        current_datetime,
        "Get the current date and time in the user's time zone.",
        ["timezone"],
        timezone={"type": "string", "description": "IANA time zone, e.g. America/New_York or UTC"},
    ),
    _tool(
        shift_datetime,
        "Add or subtract calendar days or local wall-clock minutes for a meal plan.",
        ["start"],
        start={"type": "string", "description": "YYYY-MM-DD or local YYYY-MM-DDTHH:MM without a UTC offset"},
        days={"type": "integer", "description": "Signed number of days (default 0)"},
        minutes={"type": "integer", "description": "Signed number of minutes (default 0); requires a time"},
    ),
]

TOOL_FUNCTIONS = {fn.__name__: fn for fn in (calculator, current_datetime, shift_datetime)}


def run_tool(name: str, arguments) -> str:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return "Error: unknown nutrition tool."
    if not isinstance(arguments, dict):
        return "Error: tool arguments must be an object."
    try:
        return function(**arguments)
    except TypeError:
        return "Error: missing or unexpected tool arguments."

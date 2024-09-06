from fasthtml.common import *
from fasthtml.js import HighlightJS

app, rt = fast_app(
    hdrs=[Link(rel="stylesheet", href="/static/styles.css"), HighlightJS(langs=['python', 'javascript'])],
    ftrs=[Script(src="/static/app.js")]
)


# @rt("/")
# def get():
#     return Titled("FastHTML Demo",
#                   Div(
#                       H1("Welcome to FastHTML"),
#                       P("This is a comprehensive demo of FastHTML capabilities."),
#                       Button("Click me!", hx_post="/click", hx_swap="outerHTML"),
#                       Div(id="message-container")
#                   )
#                   )

@rt("/")
def get():
    return Titled("Code Highlighting Demo",
                  H1("Python Code Example"),
                  Pre(Code("""
def hello_world():
    print("Hello, FastHTML!")

hello_world()
        """, cls="language-python")),
                  H1("JavaScript Code Example"),
                  Pre(Code("""
function greet(name) {
    console.log(`Hello, ${name}!`);
}

greet('FastHTML');
        """, cls="language-javascript"))
                  )


@rt("/click", methods=["POST"])
def post(session):
    session['clicks'] = session.get('clicks', 0) + 1
    return P(f"You've clicked {session['clicks']} times!")


@rt("/ws")
class WebSocketEndpoint(WebSocketEndpoint):
    async def on_receive(self, websocket, data):
        await websocket.send_text(f"You said: {data}")


@rt("/api/data")
def get():
    return JSONResponse({"message": "This is JSON data from the API"})


if __name__ == "__main__":
    serve()

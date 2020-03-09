from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route
from bearclassifier import BearClassifier
import sys
import uvicorn
from io import BytesIO
import json

bear_classifier = BearClassifier()

async def ping(request):
    return JSONResponse({'text':'Runing'})

async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    prediction = bear_classifier.predict(BytesIO(bytes))
    return JSONResponse({
        'prediction': prediction
    })

def home(request):
    return HTMLResponse(
        """
        <ul>
            <li> Upload image to get prediction  POST: /upload  </li>
            <li><a href="https://github.com/GeraldAburto/bear-classifier-server" target="_blank">Visit project on Github</a></li>
        </ul>
        """
    )

app = Starlette(debug=True, routes=[
    Route('/ping', ping, methods=["GET"]),
    Route('/upload', upload, methods=["POST"]),
    Route('/', home, methods=["GET"]),
])


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
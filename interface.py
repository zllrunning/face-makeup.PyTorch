from flask import Flask, request, Response, send_file

import jsonpickle
import numpy as np
import cv2
import io


from makeup import change_color
from test import evaluate
from PIL import Image

import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/demo", methods=['GET'])
def apply_makeup():
    r = request
    np_array = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    cp = "cp/79999_iter.pth"

    parsing = evaluate(img, cp)
    parsing = cv2.resize(parsing, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
    # print([key for key, value in **request.args.get().items()])
    modified_img = change_color(img, parsing, **request.args)

    cv2.imwrite("img.jpg", cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB))
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    #pil_image = Image.fromarray(modified_img)
    # pil_image = io.StringIO(Image.fromarray(modified_img))


    return send_file("img.jpg", mimetype="image/jpeg", attachment_filename="new.jpeg", as_attachment=False)



if __name__ == "__main__":
    app.debug = True
    app.run()



#
# from flask import Flask, request, Response, send_file
# import jsonpickle
# import numpy as np
# import cv2
#
# import ImageProcessingFlask
#
# # Initialize the Flask application
# app = Flask(__name__)
#
#
# # route http posts to this method
# @app.route('/api/test', methods=['POST'])
# def test():
#     r = request
#     # convert string of image data to uint8
#     nparr = np.fromstring(r.data, np.uint8)
#     # decode image
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     # do some fancy processing here....
#
#     img = ImageProcessingFlask.render(img)
#
#
#     #_, img_encoded = cv2.imencode('.jpg', img)
#     #print ( img_encoded)
#
#     cv2.imwrite( 'new.jpeg', img)
#
#
#     #response_pickled = jsonpickle.encode(response)
#     #return Response(response=response_pickled, status=200, mimetype="application/json")
#     return send_file( 'new.jpeg', mimetype="image/jpeg", attachment_filename="new.jpeg", as_attachment=True)

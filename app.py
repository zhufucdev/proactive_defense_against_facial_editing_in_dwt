import io
from typing import IO

from PIL import Image
from flask import Flask, request, abort, send_file
from torchvision import transforms

from config import no_dropout, init_type, init_gain, ngf, net_noise, norm, input_nc, output_nc, eposilon, \
    mask_model_path, img_size
from networks.DWT_model import DWT
from networks.PG_network import define_G as PG_Model
# fgan
from networks.SA_model import SA
from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.tool import *

app = Flask(__name__)

sa_net = SA(mask_model_path)  # saliency detection model
pg_model = PG_Model(input_nc, output_nc, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
dwt = DWT(img_size)


def generate_defended_image(imio: IO[bytes]):
    img = Image.open(imio)

    transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    img_tf = transform(img)
    img_t = img_tf.unsqueeze(0)
    x_real = img_t.to(device).clone().detach()
    x_ori = tensor2numpy(x_real)

    # convert RGB to YCbCr
    x_ycbcr = rgb2ycbcr_np(x_ori)
    x_y = ycbcr_to_tensor(x_ycbcr).to(device)
    x_dwt = dwt.dwt(x_y)
    x_LL, x_HL, x_LH, x_HH = dwt.get_subbands(x_dwt)
    reshape_img = dwt.dwt_to_whole(x_LL, x_HL, x_LH, x_HH)

    sa_mask = sa_net.compute_mask(x_real)

    y_mask = Y_mask(img_size)
    adv_noise = pg_model(reshape_img) * y_mask
    adv_noise = torch.clamp(adv_noise, -eposilon, eposilon)
    adv_noise = noise_clamp(adv_noise, img_size, sa_mask)

    x_L_adv = reshape_img + adv_noise
    x_adv_dwt = dwt.whole_to_dwt(x_L_adv)
    x_adv = ycbcr_to_rgb(dwt.iwt(x_adv_dwt))
    adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())
    defended_arr = adv_A[0].add_(1).mul(127.5).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    return Image.fromarray(defended_arr)


@app.route('/defend', methods=['POST'])
def defend():
    if 'file' not in request.files:
        abort(400)

    file = request.files['file']
    accept = request.headers.get('Accept', default=None)
    codec = 'jpeg'
    if accept is not None:
        paths = accept.split('/')
        if len(paths) != 2:
            abort(400)

        if paths[0] != '*' and paths[1] != '*':
            codec = paths[1]

    with file.stream as s:
        image = generate_defended_image(s)
        buf = io.BytesIO()
        image.save(buf, format=codec)
        buf.seek(0)
        return send_file(buf, mimetype=f'image/{codec}')

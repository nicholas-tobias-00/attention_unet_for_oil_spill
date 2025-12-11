import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # ---------- INPUT ----------
    to_input(r'oil spill example.png', width=12, height=12),

    # ================= ENCODER =================
    # enc1: ConvBlock 3->64
    to_Conv("enc1", 64, 64,
            offset="(1,0,0)", to="(0,0,0)",
            height=40, depth=40, width=2,
            caption="64"),
    to_Pool("pool1",
            offset="(0.5,0,0)", to="(enc1-east)",
            width=1, height=32, depth=32),

    # enc2: ConvBlock 64->128
    to_Conv("enc2", 128, 32,
            offset="(1,0,0)", to="(pool1-east)",
            height=32, depth=32, width=3,
            caption="128"),
    to_Pool("pool2",
            offset="(0.5,0,0)", to="(enc2-east)",
            width=1, height=25, depth=25),

    # enc3: ConvBlock 128->256
    to_Conv("enc3", 256, 25,
            offset="(1,0,0)", to="(pool2-east)",
            height=25, depth=25, width=4,
            caption="256"),
    to_Pool("pool3",
            offset="(0.5,0,0)", to="(enc3-east)",
            width=1, height=18, depth=18),

    # enc4: ConvBlock 256->512
    to_Conv("enc4", 512, 18,
            offset="(1,0,0)", to="(pool3-east)",
            height=18, depth=18, width=5,
            caption="512"),
    to_Pool("pool4",
            offset="(0.5,0,0)", to="(enc4-east)",
            width=1, height=12, depth=12),

    # ---------- BOTTLENECK ----------
    to_Conv("bottleneck", 1024, 12,
            offset="(1,0,0)", to="(pool4-east)",
            height=12, depth=12, width=6,
            caption="1024"),

    # ================= DECODER + AGs =================

    # ---------- LEVEL 4 ----------
    # up4: UpConv 1024->512
    to_Conv("up4", 512, 18,
            offset="(1.5,0,0)", to="(bottleneck-east)",
            height=18, depth=18, width=5,
            caption="512"),

    # att4: small AG between enc4 and up4, slightly above
    to_Conv("att4", 1, 5,
            offset="(1.2,4,0)", to="(enc4-east)",
            height=5, depth=5, width=1,
            caption="AG"),

    # polyline connections into AG4
    r"\draw [connection] (enc4-north) -- ++(0,2,0) -- (att4-west);",
    r"\draw [connection] (up4-north)  -- ++(0,1.5,0) -- (att4-east);",

    # dec4: ConvBlock 1024->512
    to_Conv("dec4", 512, 18,
            offset="(1.5,0,0)", to="(up4-east)",
            height=18, depth=18, width=5,
            caption="512"),
    to_connection("att4", "dec4"),

    # ---------- LEVEL 3 ----------
    # up3: UpConv 512->256
    to_Conv("up3", 256, 25,
            offset="(1.5,0,0)", to="(dec4-east)",
            height=25, depth=25, width=4,
            caption="256"),

    # att3 between enc3 and up3
    to_Conv("att3", 1, 5,
            offset="(1.2,6,0)", to="(enc3-east)",
            height=5, depth=5, width=1,
            caption="AG"),

    r"\draw [connection] (enc3-north) -- ++(0,2,0) -- (att3-west);",
    r"\draw [connection] (up3-north)  -- ++(0,1.5,0) -- (att3-east);",

    # dec3: ConvBlock 512->256
    to_Conv("dec3", 256, 25,
            offset="(1.5,0,0)", to="(up3-east)",
            height=25, depth=25, width=4,
            caption="256"),
    to_connection("att3", "dec3"),

    # ---------- LEVEL 2 ----------
    # up2: UpConv 256->128
    to_Conv("up2", 128, 32,
            offset="(1.5,0,0)", to="(dec3-east)",
            height=32, depth=32, width=3,
            caption="128"),

    # att2 between enc2 and up2
    to_Conv("att2", 1, 5,
            offset="(1.2,8,0)", to="(enc2-east)",
            height=5, depth=5, width=1,
            caption="AG"),

    r"\draw [connection] (enc2-north) -- ++(0,2,0) -- (att2-west);",
    r"\draw [connection] (up2-north)  -- ++(0,1.5,0) -- (att2-east);",

    # dec2: ConvBlock 256->128
    to_Conv("dec2", 128, 32,
            offset="(1.5,0,0)", to="(up2-east)",
            height=32, depth=32, width=3,
            caption="128"),
    to_connection("att2", "dec2"),

    # ---------- LEVEL 1 ----------
    # up1: UpConv 128->64
    to_Conv("up1", 64, 40,
            offset="(1.5,0,0)", to="(dec2-east)",
            height=40, depth=40, width=2,
            caption="64"),

    # att1 between enc1 and up1
    to_Conv("att1", 1, 5,
            offset="(1.2,10,0)", to="(enc1-east)",
            height=5, depth=5, width=1,
            caption="AG"),

    r"\draw [connection] (enc1-north) -- ++(0,2,0) -- (att1-west);",
    r"\draw [connection] (up1-north)  -- ++(0,1.5,0) -- (att1-east);",

    # dec1: ConvBlock 128->64
    to_Conv("dec1", 64, 40,
            offset="(1.5,0,0)", to="(up1-east)",
            height=40, depth=40, width=2,
            caption="64"),
    to_connection("att1", "dec1"),

    # ---------- OUTPUT ----------
    # 1x1 conv (64->1) + sigmoid
    to_Conv("out", 1, 40,
            offset="(1.5,0,0)", to="(dec1-east)",
            height=40, depth=40, width=1,
            caption="1"),

    to_end()
]



def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
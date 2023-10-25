import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function offsetDOMWidget(widget, ctx, node, widgetWidth, widgetY, height) {
  const margin = 10;
  const elRect = ctx.canvas.getBoundingClientRect();
  const transform = new DOMMatrix()
    .scaleSelf(
      elRect.width / ctx.canvas.width,
      elRect.height / ctx.canvas.height
    )
    .multiplySelf(ctx.getTransform())
    .translateSelf(0, widgetY + margin);

  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);
  Object.assign(widget.inputEl.style, {
    transformOrigin: "0 0",
    transform: scale,
    left: `${transform.e}px`,
    top: `${transform.d + transform.f}px`,
    width: `${widgetWidth}px`,
    height: `${(height || widget.parent?.inputHeight || 32) - margin}px`,
    position: "absolute",
    background: !node.color ? "" : node.color,
    color: !node.color ? "" : "white",
    zIndex: 5, //app.graph._nodes.indexOf(node),
  });
}

export const hasWidgets = (node) => {
  if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
    return false;
  }
  return true;
};

export const cleanupNode = (node) => {
  if (!hasWidgets(node)) {
    return;
  }

  for (const w of node.widgets) {
    if (w.canvas) {
      w.canvas.remove();
    }
    if (w.inputEl) {
      w.inputEl.remove();
    }
    // calls the widget remove callback
    w.onRemoved?.();
  }
};

export const CreatePreviewElement = (name, val, format, callback) => {
  const [type] = format.split("/");

  const w = {
    name,
    type,
    value: val,
    draw: function (ctx, node, widgetWidth, widgetY, height) {
      const [cw, ch] = this.computeSize(widgetWidth);
      offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch);
    },
    computeSize: function (_) {
      const ratio = this.inputRatio || 1;
      const width = Math.max(220, this.parent.size[0]);
      return [width, width / ratio + 10];
    },
    onRemoved: function () {
      if (this.inputEl) {
        this.inputEl.remove();
      }
    },
  };

  w.inputEl = document.createElement(type === "video" ? "video" : "img");
  w.inputEl.src = w.value;
  if (type === "video") {
    w.inputEl.setAttribute("type", "video/webm");
    w.inputEl.autoplay = true;
    w.inputEl.loop = true;
    w.inputEl.controls = false;
    w.inputEl.addEventListener("loadedmetadata", function (e) {
      w.inputRatio = this.videoWidth / this.videoHeight;
      callback?.();
    }, false);
  } else {
    w.inputEl.onload = function () {
      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight;
      callback?.();
    };
  }
  document.body.appendChild(w.inputEl);
  return w;
};

const videoPreview = {
  name: "AnimateDiff.VideoPreview",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted ? onExecuted.apply(this, message) : undefined;

      if (message?.videos) {
        this.videos = message.videos;
      }

      return r;
    };

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      const r = onDrawBackground ? onDrawBackground.apply(this, arguments) : undefined;
      const node = this;
      const prefix = "ad_video_preview_";

      if (node.videos_rendered === node.videos) {
        return r;
      }

      if (node.widgets) {
        const pos = node.widgets.findIndex((w) => w.name === `${prefix}_0`);
        if (pos !== -1) {
          for (let i = pos; i < node.widgets.length; i++) {
            node.widgets[i].onRemoved?.();
          }
          node.widgets.length = pos;
        }
      }
      if (node.videos) {
        node.videos.forEach((params, i) => {
          const previewUrl = api.apiURL(
            "/view?" + new URLSearchParams(params).toString()
          );
          const w = node.addCustomWidget(
            CreatePreviewElement(
              `${prefix}_${i}`,
              previewUrl,
              params.format || "image/gif",
              node.computeSizeKeepWidth.bind(node)
            )
          );
          w.parent = node;
        });
        node.videos_rendered = node.videos;
      }

      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      cleanupNode(this);
      return onRemoved ? onRemoved.apply(this, arguments) : undefined;
    };

    nodeType.prototype.computeSizeKeepWidth = function () {
      this.setSize([
        this.size[0],
        this.computeSize([this.size[0], this.size[1]])[1],
      ]);
    };
  },
};

app.registerExtension(videoPreview);

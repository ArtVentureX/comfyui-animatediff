import { app, ANIM_PREVIEW_WIDGET } from '../../../scripts/app.js';
import { api } from "../../../scripts/api.js";
import { $el } from '../../../scripts/ui.js';
import { createImageHost } from "../../../scripts/ui/imagePreview.js"

const URL_REGEX = /^(https?:\/\/|\/view\?|data:image\/)/;

const style = `
.comfy-img-preview video {
  object-fit: contain;
  width: var(--comfy-img-preview-width);
  height: var(--comfy-img-preview-height);
}
`;

export function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object");
    return;
  }
  if (property in object) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
};

export function formatUploadedUrl(params) {
  if (params.url) {
    return params.url;
  }

  params = { ...params };

  if (!params.filename && params.name) {
    params.filename = params.name;
    delete params.name;
  }

  return api.apiURL("/view?" + new URLSearchParams(params));
};

export function addVideoPreview(nodeType, options = {}) {
  const createVideoNode = (url) => {
    return new Promise((cb) => {
      const videoEl = document.createElement('video');
      Object.defineProperty(videoEl, 'naturalWidth', {
        get: () => {
          return videoEl.videoWidth;
        },
      });
      Object.defineProperty(videoEl, 'naturalHeight', {
        get: () => {
          return videoEl.videoHeight;
        },
      });
      videoEl.addEventListener('loadedmetadata', () => {
        videoEl.controls = false;
        videoEl.loop = true;
        videoEl.muted = true;
        cb(videoEl);
      });
      videoEl.addEventListener('error', () => {
        cb();
      });
      videoEl.src = url;
    });
  };

  const createImageNode = (url) => {
    return new Promise((cb) => {
      const imgEl = document.createElement('img');
      imgEl.onload = () => {
        cb(imgEl);
      };
      imgEl.addEventListener('error', () => {
        cb();
      });
      imgEl.src = url;
    });
  };

  nodeType.prototype.onDrawBackground = function (ctx) {
    if (this.flags.collapsed) return;

    let imageURLs = (this.images ?? []).map((i) =>
      typeof i === 'string' ? i : formatUploadedUrl(i),
    );
    let imagesChanged = false;

    if (JSON.stringify(this.displayingImages) !== JSON.stringify(imageURLs)) {
      this.displayingImages = imageURLs;
      imagesChanged = true;
    }

    if (!imagesChanged) return;
    if (!imageURLs.length) {
      this.imgs = null;
      this.animatedImages = false;
      return;
    }

    const promises = imageURLs.map((url) => {
      if (url.startsWith('/view')) {
        url = window.location.origin + url;
      }

      const u = new URL(url);
      const filename =
        u.searchParams.get('filename') || u.searchParams.get('name') || u.pathname.split('/').pop();
      const ext = filename.split('.').pop();
      const format = ['gif', 'webp', 'avif'].includes(ext) ? 'image' : 'video';
      if (format === 'video') {
        return createVideoNode(url);
      } else {
        return createImageNode(url);
      }
    });

    Promise.all(promises)
      .then((imgs) => {
        this.imgs = imgs.filter(Boolean);
      })
      .then(() => {
        if (!this.imgs.length) return;

        this.animatedImages = true;
        const widgetIdx = this.widgets?.findIndex((w) => w.name === ANIM_PREVIEW_WIDGET);

        // Instead of using the canvas we'll use a IMG
        if (widgetIdx > -1) {
          // Replace content
          const widget = this.widgets[widgetIdx];
          widget.options.host.updateImages(this.imgs);
        } else {
          const host = createImageHost(this);
          this.setSizeForImage(true);
          const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', host.el, {
            host,
            getHeight: host.getHeight,
            onDraw: host.onDraw,
            hideOnZoom: false,
          });
          widget.serializeValue = () => ({
            height: host.el.clientHeight,
          });
          // widget.computeSize = (w) => ([w, 220]);

          widget.options.host.updateImages(this.imgs);
        }

        this.imgs.forEach((img) => {
          if (img instanceof HTMLVideoElement) {
            img.muted = true;
            img.autoplay = true;
            img.play();
          }
        });
      });
  };

  const { textWidget, comboWidget } = options;

  if (textWidget) {
    chainCallback(nodeType.prototype, 'onNodeCreated', function () {
      const pathWidget = this.widgets.find((w) => w.name === textWidget);
      pathWidget._value = pathWidget.value;
      Object.defineProperty(pathWidget, 'value', {
        set: (value) => {
          pathWidget._value = value;
          pathWidget.inputEl.value = value;
          this.images = (value ?? '').split('\n').filter((url) => URL_REGEX.test(url));
        },
        get: () => {
          return pathWidget._value;
        },
      });
      pathWidget.inputEl.addEventListener('change', (e) => {
        const value = e.target.value;
        pathWidget._value = value;
        this.images = (value ?? '').split('\n').filter((url) => URL_REGEX.test(url));
      });

      // Set value to ensure preview displays on initial add.
      pathWidget.value = pathWidget._value;
    });
  }

  if (comboWidget) {
    chainCallback(nodeType.prototype, 'onNodeCreated', function () {
      const pathWidget = this.widgets.find((w) => w.name === comboWidget);
      pathWidget._value = pathWidget.value;
      Object.defineProperty(pathWidget, 'value', {
        set: (value) => {
          pathWidget._value = value;
          if (!value) {
            return this.images = []
          }

          const parts = value.split("/")
          const filename = parts.pop()
          const subfolder = parts.join("/")
          const extension = filename.split(".").pop();
          const format = (["gif", "webp", "avif"].includes(extension)) ? 'image' : 'video'
          this.images = [formatUploadedUrl({ filename, subfolder, type: "input", format: format })]
        },
        get: () => {
          return pathWidget._value;
        },
      });

      // Set value to ensure preview displays on initial add.
      pathWidget.value = pathWidget._value;
    });
  }

  chainCallback(nodeType.prototype, "onExecuted", function (message) {
    if (message?.videos) {
      this.images = message?.videos.map(formatUploadedUrl);
    }
  });
}

app.registerExtension({
  name: "AnimateDiff.VideoPreview",
  init() {
    $el('style', {
      textContent: style,
      parent: document.head,
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AnimateDiffCombine") {
      return;
    }

    addVideoPreview(nodeType);
  },
});

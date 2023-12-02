import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

export const chainCallback = (object, property, callback) => {
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

export const formatUploadedUrl = (params) => {
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

export const fitHeight = (node) => {
  node.setSize([
    node.size[0],
    node.computeSize([node.size[0], node.size[1]])[1],
  ]);
  node.graph.setDirtyCanvas(true);
};

export const addVideoPreview = (nodeType) => {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    let previewNode = this;
    //preview is a made up widget type to enable user defined functions
    //videopreview is widget name
    //The previous implementation used type to distinguish between a video and gif,
    //but the type is not serialized and would not survive a reload
    var previewWidget = {
      name: "videopreview",
      type: "preview",
      value: "",
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        //update widget position, hide if off-screen
        const transform = ctx.getTransform();
        const scale = app.canvas.ds.scale; //gets the litegraph zoom
        //calculate coordinates with account for browser zoom
        const x = (transform.e * scale) / transform.a;
        const y = (transform.f * scale) / transform.a;
        Object.assign(this.parentEl.style, {
          left: x + 15 * scale + "px",
          top: y + widgetY * scale + "px",
          width: (widgetWidth - 30) * scale + "px",
          zIndex: 2 + (node.is_selected ? 1 : 0),
          position: "absolute",
        });
        this._boundingCount = 0;
      },
      computeSize: function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
          let height = (previewNode.size[0] - 30) / this.aspectRatio;
          if (!(height > 0)) {
            height = 0;
          }
          return [width, height];
        }
        return [width, -4]; //no loaded src, widget should not display
      },
      _value: { hidden: false, paused: false },
    };
    //onRemoved isn't a litegraph supported function on widgets
    //Given that onremoved widget and node callbacks are sparse, this
    //saves the required iteration.
    chainCallback(this, "onRemoved", () => {
      previewWidget?.parentEl?.remove();
    });
    this.addCustomWidget(previewWidget);
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "animatediff_preview";
    previewWidget.parentEl.style["pointer-events"] = "none";

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = false;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true;
    previewWidget.videoEl.style["width"] = "100%";
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
      previewWidget.aspectRatio =
        previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
      fitHeight(this);
    });
    previewWidget.videoEl.addEventListener("error", () => {
      //TODO: consider a way to properly notify the user why a preview isn't shown.
      previewWidget.parentEl.hidden = true;
      fitHeight(this);
    });

    previewWidget.imgEl = document.createElement("img");
    previewWidget.imgEl.style["width"] = "100%";
    previewWidget.imgEl.hidden = true;
    previewWidget.imgEl.onload = () => {
      previewWidget.aspectRatio =
        previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
      fitHeight(this);
    };

    this.setPreviewsrc = (params) => {
      previewWidget._value = params;
      this._setPreviewsrc(params);
    };
    this._setPreviewsrc = function (params) {
      if (params == undefined) {
        return;
      }
      previewWidget.parentEl.hidden = previewWidget._value.hidden;
      if (params?.format?.split("/")[0] == "video") {
        previewWidget.videoEl.autoplay =
          !previewWidget._value.paused && !previewWidget._value.hidden;
        previewWidget.videoEl.src = formatUploadedUrl(params);
        previewWidget.videoEl.hidden = false;
        previewWidget.imgEl.hidden = true;
      } else {
        // Is animated image
        previewWidget.imgEl.src = formatUploadedUrl(params);
        previewWidget.videoEl.hidden = true;
        previewWidget.imgEl.hidden = false;
      }
    };
    Object.defineProperty(previewWidget, "value", {
      set: (value) => {
        if (value) {
          previewWidget._value = value;
          this._setPreviewsrc(value.params);
        }
      },
      get: () => {
        return previewWidget._value;
      },
    });
    //Hide video element if offscreen
    //The multiline input implementation moves offscreen every frame
    //and doesn't apply until a node with an actual inputEl is loaded
    this._boundingCount = 0;
    this.onBounding = function () {
      if (this._boundingCount++ > 5) {
        previewWidget.parentEl.style.left = "-8000px";
      }
    };
    previewWidget.parentEl.appendChild(previewWidget.videoEl);
    previewWidget.parentEl.appendChild(previewWidget.imgEl);
    document.body.appendChild(previewWidget.parentEl);
  });
};

export function addVideoPreviewOptions(nodeType) {
  chainCallback(
    nodeType.prototype,
    "getExtraMenuOptions",
    function (_, options) {
      // The intended way of appending options is returning a list of extra options,
      // but this isn't used in widgetInputs.js and would require
      // less generalization of chainCallback
      let optNew = [];
      const previewWidget = this.widgets.find((w) => w.name === "videopreview");

      let url = null;
      if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
        url = previewWidget.videoEl.src;
      } else if (
        previewWidget.imgEl?.hidden == false &&
        previewWidget.imgEl.src
      ) {
        url = previewWidget.imgEl.src;
      }
      if (url) {
        url = new URL(url);
        //placeholder from Save Image, will matter once preview functionality is implemented
        //url.searchParams.delete('preview')
        optNew.push(
          {
            content: "Open preview",
            callback: () => {
              window.open(url, "_blank");
            },
          },
          {
            content: "Save preview",
            callback: () => {
              const a = document.createElement("a");
              a.href = url;
              a.setAttribute(
                "download",
                new URLSearchParams(url.search).get("filename")
              );
              document.body.append(a);
              a.click();
              requestAnimationFrame(() => a.remove());
            },
          }
        );
      }
      const PauseDesc =
        (previewWidget._value.paused ? "Resume" : "Pause") + " preview";
      if (previewWidget.videoEl.hidden == false) {
        optNew.push({
          content: PauseDesc,
          callback: () => {
            //animated images can't be paused and are more likely to cause performance issues.
            //changing src to a single keyframe is possible,
            //For now, the option is disabled if an animated image is being displayed
            if (previewWidget._value.paused) {
              previewWidget.videoEl?.play();
            } else {
              previewWidget.videoEl?.pause();
            }
            previewWidget._value.paused = !previewWidget._value.paused;
          },
        });
      }
      //TODO: Consider hiding elements if video no preview is available yet.
      //It would reduce confusion at the cost of functionality
      //(if a video preview lags the computer, the user should be able to hide in advance)
      const visDesc =
        (previewWidget._value.hidden ? "Show" : "Hide") + " preview";
      optNew.push({
        content: visDesc,
        callback: () => {
          if (!previewWidget.videoEl.hidden && !previewWidget._value.hidden) {
            previewWidget.videoEl.pause();
          } else if (
            previewWidget._value.hidden &&
            !previewWidget.videoEl.hidden &&
            !previewWidget._value.paused
          ) {
            previewWidget.videoEl.play();
          }
          previewWidget._value.hidden = !previewWidget._value.hidden;
          previewWidget.parentEl.hidden = previewWidget._value.hidden;
          fitHeight(this);
        },
      });
      optNew.push({
        content: "Sync preview",
        callback: () => {
          //TODO: address case where videos have varying length
          //Consider a system of sync groups which are opt-in?
          for (let p of document.getElementsByClassName("vhs_preview")) {
            for (let child of p.children) {
              if (child.tagName == "VIDEO") {
                child.currentTime = 0;
              } else if (child.tagName == "IMG") {
                child.src = child.src;
              }
            }
          }
        },
      });
      if (options.length > 0 && options[0] != null && optNew.length > 0) {
        optNew.push(null);
      }
      options.unshift(...optNew);
    }
  );
}

app.registerExtension({
  name: "AnimateDiff.VideoPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AnimateDiffCombine") {
      return;
    }

    addVideoPreview(nodeType);
    addVideoPreviewOptions(nodeType);
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      this._outputs = this.outputs;
      Object.defineProperty(this, "outputs", {
        set: function (value) {
          this._outputs = value;
          requestAnimationFrame(() => {
            if (app.nodeOutputs[this.id + ""]) {
              this.setPreviewsrc(app.nodeOutputs[this.id + ""].videos[0]);
            }
          });
        },
        get: function () {
          return [];
        },
      });
    });
    chainCallback(nodeType.prototype, "onExecuted", function (message) {
      if (message?.videos) {
        this.setPreviewsrc(message.videos[0]);
      }
    });
  },
});

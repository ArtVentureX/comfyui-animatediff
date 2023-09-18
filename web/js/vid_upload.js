import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const supportedVideoTypes = [
  "image/gif",
  "video/webm",
  "video/mp4",
  "video/mov",
];

const VIDEOUPLOAD = (node, inputName, inputData, app) => {
  const previewWidget = "ad_video_preview";
  const videoWidget = node.widgets.find((w) => w.name === "video");
  let uploadWidget;

  const showVideo = (name) => {
    let folder_separator = name.lastIndexOf("/");
    let subfolder = "";
    if (folder_separator > -1) {
      subfolder = name.substring(0, folder_separator);
      name = name.substring(folder_separator + 1);
    }
    const ext = name.substring(name.lastIndexOf(".") + 1);
    const format = supportedVideoTypes.find((t) => t.endsWith(ext));
    node.videos = [
      {
        filename: name,
        type: "input",
        subfolder: subfolder,
        format,
      },
    ];
  };

  var default_value = videoWidget.value;
  Object.defineProperty(videoWidget, "value", {
    set: function (value) {
      this._real_value = value;
    },

    get: function () {
      let value = "";
      if (this._real_value) {
        value = this._real_value;
      } else {
        return default_value;
      }

      if (value.filename) {
        let real_value = value;
        value = "";
        if (real_value.subfolder) {
          value = real_value.subfolder + "/";
        }

        value += real_value.filename;

        if (real_value.type && real_value.type !== "input")
          value += ` [${real_value.type}]`;
      }
      return value;
    },
  });

  // Add our own callback to the combo widget to render an image when it changes
  const cb = node.callback;
  videoWidget.callback = function () {
    showVideo(videoWidget.value);
    if (cb) {
      return cb.apply(this, arguments);
    }
  };

  // On load if we have a value then render the image
  // The value isnt set immediately so we need to wait a moment
  // No change callbacks seem to be fired on initial setting of the value
  requestAnimationFrame(() => {
    if (videoWidget.value) {
      showVideo(videoWidget.value);
    }
  });

  async function uploadFile(file, updateNode, pasted = false) {
    try {
      // Wrap file in formdata so it includes filename
      const body = new FormData();
      body.append("image", file);
      body.append("subfolder", "video");
      const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
      });

      if (resp.status === 200) {
        const data = await resp.json();
        // Add the file to the dropdown list and update the widget value
        let path = data.name;
        if (data.subfolder) path = data.subfolder + "/" + path;

        if (!videoWidget.options.values.includes(path)) {
          videoWidget.options.values.push(path);
        }

        if (updateNode) {
          showVideo(path);
          videoWidget.value = path;
        }
      } else {
        alert(resp.status + " - " + resp.statusText);
      }
    } catch (error) {
      alert(error);
    }
  }

  const fileInput = document.createElement("input");
  Object.assign(fileInput, {
    type: "file",
    accept: supportedVideoTypes.join(","),
    style: "display: none",
    onchange: async () => {
      if (fileInput.files.length) {
        await uploadFile(fileInput.files[0], true);
      }
    },
  });
  document.body.append(fileInput);

  // Create the button widget for selecting the files
  uploadWidget = node.addWidget(
    "button",
    "choose file to upload",
    "image",
    () => {
      fileInput.click();
    }
  );
  uploadWidget.serialize = false;

  // Add handler to check if an image is being dragged over our node
  node.onDragOver = function (e) {
    if (e.dataTransfer && e.dataTransfer.items) {
      const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
      return !!image;
    }

    return false;
  };

  // On drop upload files
  node.onDragDrop = function (e) {
    console.log("onDragDrop called");
    let handled = false;
    for (const file of e.dataTransfer.files) {
      if (file.type.startsWith("image/")) {
        uploadFile(file, !handled); // Dont await these, any order is fine, only update on first one
        handled = true;
      }
    }

    return handled;
  };

  node.pasteFile = function (file) {
    if (supportedVideoTypes.indexOf(file.type) > -1) {
      const is_pasted =
        file.name === "image.png" && file.lastModified - Date.now() < 2000;
      uploadFile(file, true, is_pasted);
      return true;
    }
    return false;
  };

  return { widget: uploadWidget };
};

ComfyWidgets["VIDEOUPLOAD"] = VIDEOUPLOAD;

// Adds an upload button to the nodes
app.registerExtension({
  name: "AnimateDiff.UploadVideo",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.input?.required?.video?.[1]?.video_upload === true) {
      nodeData.input.required.upload = ["VIDEOUPLOAD"];
    }
  },
});

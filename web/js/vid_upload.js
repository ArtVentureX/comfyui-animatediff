import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

import {
  chainCallback,
  addVideoPreview,
} from "./vid_preview.js";

async function uploadFile(file) {
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData();
    const new_file = new File([file], file.name, {
      type: file.type,
      lastModified: file.lastModified,
    });
    body.append("image", new_file);
    body.append("subfolder", "video");
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body,
    });

    if (resp.status === 200 || resp.status === 201) {
      return resp.json();
    } else {
      alert(`Upload failed: ${resp.statusText}`);
    }
  } catch (error) {
    alert(`Upload failed: ${error}`);
  }
}

function addUploadWidget(nodeType, widgetName) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const pathWidget = this.widgets.find((w) => w.name === widgetName);
    if (pathWidget.element) {
      pathWidget.options.getMinHeight = () => 50;
      pathWidget.options.getMaxHeight = () => 150;
    }

    const fileInput = document.createElement("input");
    chainCallback(this, "onRemoved", () => {
      fileInput?.remove();
    });

    Object.assign(fileInput, {
      type: "file",
      accept: "video/webm,video/mp4,video/mkv,image/gif,image/webp",
      style: "display: none",
      onchange: async () => {
        if (fileInput.files.length) {
          const params = await uploadFile(fileInput.files[0]);
          if (!params) {
            // upload failed and file can not be added to options
            return;
          }

          fileInput.value = "";
          const filename = [params.subfolder, params.name || params.filename].filter(Boolean).join('/')
          pathWidget.value = filename;
          pathWidget.options.values.push(filename);
        }
      },
    });

    document.body.append(fileInput);
    let uploadWidget = this.addWidget(
      "button",
      "choose video to upload",
      "image",
      () => {
        app.canvas.node_widget = null;
        fileInput.click();
      }
    );
    uploadWidget.options.serialize = false;
  });
}

// Adds an upload button to the nodes
app.registerExtension({
  name: "AnimateDiff.UploadVideo",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.input?.required?.video?.[1]?.video_upload === true) {
      addUploadWidget(nodeType, 'video');
      addVideoPreview(nodeType, { comboWidget: 'video' });
    }
  },
});

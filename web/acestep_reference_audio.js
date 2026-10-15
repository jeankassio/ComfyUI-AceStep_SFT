import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AceStepSFT.ReferenceAudio",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AceStepSFTLoadReferenceAudio") return;

        const inputs = nodeData.input.required;
        // The native uploader needs the audio preview widget to exist first.
        const upload = inputs.upload;
        delete inputs.upload;
        inputs.audioUI = ["AUDIO_UI", {}];
        if (upload) inputs.upload = upload;
    },
});

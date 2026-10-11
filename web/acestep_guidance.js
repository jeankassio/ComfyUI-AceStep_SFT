import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AceStepSFT.GuidanceSchedule",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["AceStepSFTGenerate", "AceStepSFTGenerateAdvanced"].includes(nodeData.name)) return;
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = originalConfigure?.apply(this, arguments);
            const widget = this.widgets?.find((item) => item.name === "guidance_schedule");
            if (!widget) return result;
            const value = info.widgets_values?.[this.widgets.indexOf(widget)] ?? widget.value;
            if (typeof value === "number" && Number.isFinite(value) && value >= -1 && value <= 1) {
                widget.value = value === 0 ? "normal" : value > 0 ? "decay" : "ramp_up";
            }
            return result;
        };
    },
});

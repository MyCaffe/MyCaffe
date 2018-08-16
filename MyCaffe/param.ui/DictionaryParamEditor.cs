using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.Design;

namespace MyCaffe.param.ui
{
    /// <summary>
    /// The DictionaryParamEditor is used to visually edit dictionary based parameters that are stored as a key/value set packed within a string.
    /// </summary>
    public class DictionaryParamEditor : UITypeEditor
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public DictionaryParamEditor() : base()
        {
        }

        /// <summary>
        /// Returns the edit style DROPDOWN
        /// </summary>
        /// <param name="context">Specifies the context (not used).</param>
        /// <returns></returns>
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context)
        {
            return UITypeEditorEditStyle.DropDown;
        }

        /// <summary>
        /// The EditValue displays the editing control and returns the new edited value.
        /// </summary>
        /// <param name="context">Specifies the context (not used).</param>
        /// <param name="provider">Specifies the provider implementing the service.</param>
        /// <param name="value">Specifies the value to edit.</param>
        /// <returns>The edited value is returned.</returns>
        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            IWindowsFormsEditorService svc = null;

            if (provider != null)
                svc = provider.GetService(typeof(IWindowsFormsEditorService)) as IWindowsFormsEditorService;

            if (svc != null)
            {
                DictionaryParameterEditorControl ctrl = new DictionaryParameterEditorControl((string)value, svc);
                svc.DropDownControl(ctrl);
                value = ctrl.Value;
            }

            return value;
        }
    }
}
